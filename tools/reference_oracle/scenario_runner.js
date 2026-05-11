#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { EventEmitter } = require('events');

const repoRoot = path.resolve(__dirname, '..', '..');
const referenceRoot = path.join(repoRoot, 'third_party', 'curvytron-reference');
const defaultFixturePath = path.join(
  repoRoot,
  'scenarios',
  'environment',
  'forced_two_player_turn_step.json'
);

const sourceFiles = [
  'src/shared/Collection.js',
  'src/shared/service/BaseFPSLogger.js',
  'src/server/service/FPSLogger.js',
  'src/shared/service/Compressor.js',
  'src/shared/model/BaseBonus.js',
  'src/server/model/Bonus/Bonus.js',
  'src/server/model/Bonus/BonusSelf.js',
  'src/server/model/Bonus/BonusSelfSmall.js',
  'src/server/model/Bonus/BonusSelfSlow.js',
  'src/server/model/Bonus/BonusSelfFast.js',
  'src/server/model/Bonus/BonusSelfMaster.js',
  'src/server/model/Bonus/BonusEnemy.js',
  'src/server/model/Bonus/BonusEnemySlow.js',
  'src/server/model/Bonus/BonusEnemyFast.js',
  'src/server/model/Bonus/BonusEnemyBig.js',
  'src/server/model/Bonus/BonusEnemyInverse.js',
  'src/server/model/Bonus/BonusEnemyStraightAngle.js',
  'src/server/model/Bonus/BonusGame.js',
  'src/server/model/Bonus/BonusGameBorderless.js',
  'src/server/model/Bonus/BonusGameClear.js',
  'src/server/model/Bonus/BonusAll.js',
  'src/server/model/Bonus/BonusAllColor.js',
  'src/shared/model/BaseBonusStack.js',
  'src/shared/model/BaseTrail.js',
  'src/server/model/Trail.js',
  'src/shared/model/BaseAvatar.js',
  'src/server/model/BonusStack.js',
  'src/shared/model/BasePlayer.js',
  'src/shared/manager/BaseBonusManager.js',
  'src/server/manager/BonusManager.js',
  'src/server/manager/PrintManager.js',
  'src/server/core/Body.js',
  'src/server/core/Island.js',
  'src/server/core/World.js',
  'src/server/core/AvatarBody.js',
  'src/server/core/SocketGroup.js',
  'src/server/controller/GameController.js',
  'src/server/model/GameBonusStack.js',
  'src/shared/model/BaseGame.js',
  'src/server/model/Avatar.js',
  'src/server/model/Player.js',
  'src/server/model/Game.js',
];

function relative(filePath) {
  return path.relative(repoRoot, filePath);
}

function hasOwn(object, key) {
  return Object.prototype.hasOwnProperty.call(object, key);
}

function firstOwned(object, keys) {
  if (!object || typeof object !== 'object') {
    return undefined;
  }
  for (const key of keys) {
    if (hasOwn(object, key)) {
      return object[key];
    }
  }
  return undefined;
}

function requiredRandomValue(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value >= 1) {
    throw new Error(`${field} must be a finite number in [0, 1)`);
  }
  return value;
}

function randomPolicy(scenario) {
  const sourceSetup = scenario && scenario.source_setup;
  const random = sourceSetup && sourceSetup.random;
  const rawSequence = firstOwned(random, [
    'math_random_sequence',
    'mathRandomSequence',
    'math_random_tape',
    'mathRandomTape',
  ]);

  if (typeof rawSequence !== 'undefined') {
    if (!Array.isArray(rawSequence)) {
      throw new Error('source_setup.random.math_random_sequence must be an array');
    }
    const labels = [];
    return {
      sequence: rawSequence.map((item, index) => {
        if (item && typeof item === 'object' && !Array.isArray(item)) {
          labels[index] = typeof item.label === 'string' ? item.label : null;
          return requiredRandomValue(
            item.value,
            `source_setup.random.math_random_sequence[${index}].value`
          );
        }
        labels[index] = null;
        return requiredRandomValue(
          item,
          `source_setup.random.math_random_sequence[${index}]`
        );
      }),
      labels,
      constant: null,
    };
  }

  return {
    sequence: null,
    labels: null,
    constant: requiredRandomValue(
      firstOwned(random, ['math_random', 'mathRandom']) ?? 0.5,
      'source_setup.random.math_random'
    ),
  };
}

function makeDeterministicMath(scenario, randomCalls) {
  const policy = randomPolicy(scenario);
  const math = Object.create(Math);
  let index = 0;

  math.random = function random() {
    let value;
    if (policy.sequence) {
      if (index >= policy.sequence.length) {
        throw new Error(`Math.random tape exhausted after ${index} calls`);
      }
      value = policy.sequence[index];
    } else {
      value = policy.constant;
    }
    const call = { index, value };
    if (policy.labels && policy.labels[index]) {
      call.label = policy.labels[index];
    }
    randomCalls.push(call);
    index += 1;
    return value;
  };
  return math;
}

function makeContext(scenario) {
  const timers = [];
  const randomCalls = [];
  let nowMs = 0;
  let nextTimerId = 1;
  let nextTimerOrder = 1;

  function callbackName(callback) {
    return callback && callback.name ? callback.name : 'anonymous';
  }

  function normalizedDelay(delay) {
    const value = Number(delay);
    return Number.isFinite(value) && value > 0 ? value : 0;
  }

  function scheduleTimer(callback, delay, interval) {
    const timer = {
      type: interval ? 'interval' : 'timeout',
      id: nextTimerId++,
      delay: normalizedDelay(delay),
      dueAt: nowMs + normalizedDelay(delay),
      order: nextTimerOrder++,
      callback,
      callbackName: callbackName(callback),
      active: true,
      interval,
    };
    timers.push(timer);
    return timer;
  }

  function clearTimer(timer) {
    if (timer && typeof timer === 'object') {
      timer.active = false;
    }
    return null;
  }

  function nextDueTimer(targetMs) {
    let selected = null;
    for (const timer of timers) {
      if (!timer.active || timer.dueAt > targetMs) {
        continue;
      }
      if (
        !selected ||
        timer.dueAt < selected.dueAt ||
        (timer.dueAt === selected.dueAt && timer.order < selected.order)
      ) {
        selected = timer;
      }
    }
    return selected;
  }

  function advanceTimers(ms) {
    const advanceMs = Number(ms);
    if (!Number.isFinite(advanceMs) || advanceMs < 0) {
      throw new Error('timer advance must be a non-negative finite number');
    }

    const targetMs = nowMs + advanceMs;
    let guard = 0;
    let timer = nextDueTimer(targetMs);
    while (timer) {
      if (guard++ > 100000) {
        throw new Error('timer advance exceeded callback guard');
      }
      nowMs = timer.dueAt;
      if (!timer.interval) {
        timer.active = false;
      }
      if (typeof timer.callback === 'function') {
        timer.callback();
      }
      if (timer.interval && timer.active) {
        timer.dueAt += timer.delay;
        timer.order = nextTimerOrder++;
      }
      timer = nextDueTimer(targetMs);
    }
    nowMs = targetMs;
    return nowMs;
  }

  return vm.createContext({
    console,
    EventEmitter,
    Math: makeDeterministicMath(scenario, randomCalls),
    Date,
    JSON,
    setTimeout(callback, delay) {
      return scheduleTimer(callback, delay, false);
    },
    clearTimeout(timer) {
      return clearTimer(timer);
    },
    setInterval(callback, delay) {
      return scheduleTimer(callback, delay, true);
    },
    clearInterval(timer) {
      return clearTimer(timer);
    },
    __advanceTimers: advanceTimers,
    __randomCalls: randomCalls,
  });
}

function loadOriginalSources(context) {
  for (const file of sourceFiles) {
    const absolute = path.join(referenceRoot, file);
    const code = fs.readFileSync(absolute, 'utf8');
    vm.runInContext(code, context, { filename: file });
  }
}

function defaultScenario() {
  return {
    id: 'forced_two_player_turn_step',
    step_ms: 1000 / 60,
    players: [
      {
        id: 'p0',
        avatar_id: 1,
        name: 'p0',
        color: '#ff0000',
        state: { x: 20, y: 40, angle: 0, printing: true },
      },
      {
        id: 'p1',
        avatar_id: 2,
        name: 'p1',
        color: '#00ff00',
        state: { x: 60, y: 40, angle: Math.PI, printing: true },
      },
    ],
    ticks: [
      { moves: { p0: -1, p1: 1 } },
    ],
  };
}

function readScenario() {
  const requested = process.argv[2];
  const fixture = requested ? path.resolve(requested) : defaultFixturePath;

  if (fs.existsSync(fixture)) {
    return {
      source: relative(fixture),
      scenario: JSON.parse(fs.readFileSync(fixture, 'utf8')),
    };
  }

  if (requested) {
    throw new Error(`Scenario file not found: ${requested}`);
  }

  return {
    source: 'builtin:forced_two_player_turn_step',
    scenario: defaultScenario(),
  };
}

function runScenario(context, scenario) {
  context.__scenario = scenario;

  return vm.runInContext(`
(function (scenario) {
  var events = [];

  function round(value) {
    return Math.round(value * 1000000) / 1000000;
  }

  function numberOr(value, fallback) {
    return typeof value === 'number' && isFinite(value) ? value : fallback;
  }

  function requiredNumber(value, field) {
    if (typeof value !== 'number' || !isFinite(value)) {
      throw new Error(field + ' must be a finite number');
    }
    return value;
  }

  function optionalNonNegativeNumber(value, field) {
    if (typeof value === 'undefined' || value === null) {
      return null;
    }
    if (typeof value !== 'number' || !isFinite(value) || value < 0) {
      throw new Error(field + ' must be a non-negative finite number');
    }
    return value;
  }

  function requiredInteger(value, field) {
    if (typeof value !== 'number' || !isFinite(value) || Math.floor(value) !== value) {
      throw new Error(field + ' must be an integer');
    }
    return value;
  }

  function requiredString(value, field) {
    if (typeof value !== 'string' || !value) {
      throw new Error(field + ' must be a non-empty string');
    }
    return value;
  }

  function avatarId(avatar) {
    return avatar ? avatar.id : null;
  }

  function bonusEffectsData(bonus) {
    if (!bonus || !bonus.target || typeof bonus.getEffects !== 'function') {
      return null;
    }
    var effects = bonus.getEffects(bonus.target);
    if (!Array.isArray(effects)) {
      return null;
    }
    return effects.map(function (effect) {
      return [effect[0], effect[1]];
    });
  }

  function bonusData(bonus) {
    return bonus ? {
      id: bonus.id,
      type: bonus.constructor.name,
      duration: typeof bonus.duration === 'undefined' ? null : bonus.duration,
      effects: bonusEffectsData(bonus)
    } : null;
  }

  function eventData(name, data) {
    switch (name) {
      case 'position':
        return { avatar: data.id, x: round(data.x), y: round(data.y) };
      case 'angle':
        return { avatar: data.id, angle: round(data.angle) };
      case 'point':
        return {
          avatar: data.avatar.id,
          x: round(data.x),
          y: round(data.y),
          important: data.important ? true : false
        };
      case 'die':
        return {
          avatar: data.avatar.id,
          killer: data.killer ? data.killer.id : null,
          old: typeof data.old === 'undefined' ? null : data.old
        };
      case 'score':
      case 'score:round':
        return { avatar: data.id, score: data.score, roundScore: data.roundScore };
      case 'property':
        return { avatar: data.avatar.id, property: data.property, value: data.value };
      case 'bonus:pop':
        return { bonus: data.id, type: data.constructor.name, x: round(data.x), y: round(data.y) };
      case 'bonus:clear':
        return { bonus: data.id };
      case 'bonus:stack':
        return {
          avatar: data.avatar.id,
          method: data.method,
          bonus: bonusData(data.bonus)
        };
      case 'round:end':
        return { winner: avatarId(data.winner) };
      case 'borderless':
        return { value: data ? true : false };
      default:
        return {};
    }
  }

  function record(emitter, name, alias) {
    emitter.on(name, function (data) {
      events.push({ event: alias || name, data: eventData(alias || name, data || {}) });
    });
  }

  function snapshotAvatar(avatar, move) {
    var lastPoint = avatar.trail.points.length
      ? avatar.trail.points[avatar.trail.points.length - 1]
      : null;

    return {
      id: avatar.id,
      name: avatar.name,
      move: move,
      x: round(avatar.x),
      y: round(avatar.y),
      angle: round(avatar.angle),
      velocity: round(avatar.velocity),
      velocityX: round(avatar.velocityX),
      velocityY: round(avatar.velocityY),
      angularVelocity: round(avatar.angularVelocity),
      radius: round(avatar.radius),
      alive: avatar.alive,
      present: avatar.present,
      printing: avatar.printing,
      score: avatar.score,
      roundScore: avatar.roundScore,
      trailPointCount: avatar.trail.points.length,
      lastTrailPoint: lastPoint ? [round(lastPoint[0]), round(lastPoint[1])] : null,
      bodyNum: avatar.body.num,
      bodyCount: avatar.bodyCount,
      activeBonuses: avatar.bonusStack.bonuses.items.map(function (bonus) {
        return bonusData(bonus);
      }),
      printManager: {
        active: avatar.printManager.active,
        distance: round(avatar.printManager.distance),
        lastX: round(avatar.printManager.lastX),
        lastY: round(avatar.printManager.lastY)
      }
    };
  }

  function snapshotGame(game) {
    return {
      size: game.size,
      started: game.started,
      inRound: game.inRound,
      borderless: game.borderless,
      deathCount: game.deaths.count(),
      deaths: game.deaths.items.map(function (avatar) { return avatar.id; }),
      roundWinner: avatarId(game.roundWinner),
      gameWinner: avatarId(game.gameWinner),
      worldActive: game.world.active,
      worldBodyCount: game.world.bodyCount,
      bonusCount: game.bonusManager.bonuses.count(),
      bonusWorldBodyCount: game.bonusManager.world.bodyCount
    };
  }

  function stateFor(playerSpec) {
    return playerSpec.state || playerSpec.initial || playerSpec;
  }

  function stateAngle(state, fallback) {
    return numberOr(state.angle, numberOr(state.angle_rad, fallback));
  }

  function ownProperty(object, key) {
    return Object.prototype.hasOwnProperty.call(object, key);
  }

  function firstDefined(object, keys) {
    if (!object || typeof object !== 'object') {
      return undefined;
    }
    for (var i = 0; i < keys.length; i++) {
      if (ownProperty(object, keys[i])) {
        return object[keys[i]];
      }
    }
    return undefined;
  }

  function forcedStateValue(playerSpec, keys) {
    var state = stateFor(playerSpec);
    var value = firstDefined(state, keys);
    return typeof value === 'undefined' ? firstDefined(playerSpec, keys) : value;
  }

  function optionalInteger(value, field) {
    return typeof value === 'undefined' || value === null ? null : requiredInteger(value, field);
  }

  function hasForcedPrintRuntimeState(state) {
    return state && typeof state === 'object' && (
      ownProperty(state, 'print_manager') ||
      ownProperty(state, 'printManager') ||
      ownProperty(state, 'trail')
    );
  }

  function applyForcedState(avatar, playerSpec) {
    var state = stateFor(playerSpec);

    avatar.present = typeof state.present === 'boolean' ? state.present : true;
    avatar.alive = typeof state.alive === 'boolean' ? state.alive : true;

    if (typeof state.score === 'number') {
      avatar.setScore(state.score);
    }

    if (typeof state.roundScore === 'number') {
      avatar.setRoundScore(state.roundScore);
    }

    if (typeof state.radius === 'number') {
      avatar.setRadius(state.radius);
    }

    if (typeof state.velocity === 'number') {
      avatar.setVelocity(state.velocity);
    }

    avatar.setPosition(numberOr(state.x, avatar.x), numberOr(state.y, avatar.y));
    avatar.setAngle(stateAngle(state, avatar.angle));
    avatar.updateVelocities();

    if (hasForcedPrintRuntimeState(state)) {
      if (typeof state.printing === 'boolean') {
        avatar.printing = state.printing;
      }
    } else if (state.printing === false) {
      avatar.printManager.stop();
    } else {
      avatar.printManager.start();
    }
  }

  function applyForcedPrintManagerState(avatar, playerSpec, index) {
    var state = stateFor(playerSpec);
    var printManager = firstDefined(state, ['print_manager', 'printManager']);
    var lastX;
    var lastY;

    if (typeof printManager === 'undefined') {
      return;
    }
    if (!printManager || typeof printManager !== 'object') {
      throw new Error('players[' + index + '].initial.print_manager must be an object');
    }

    if (ownProperty(printManager, 'active')) {
      if (typeof printManager.active !== 'boolean') {
        throw new Error('players[' + index + '].initial.print_manager.active must be a boolean');
      }
      avatar.printManager.active = printManager.active;
    }
    if (ownProperty(printManager, 'distance')) {
      avatar.printManager.distance = requiredNumber(
        printManager.distance,
        'players[' + index + '].initial.print_manager.distance'
      );
    }
    lastX = firstDefined(printManager, ['last_x', 'lastX']);
    if (typeof lastX !== 'undefined') {
      avatar.printManager.lastX = requiredNumber(
        lastX,
        ownProperty(printManager, 'last_x')
          ? 'players[' + index + '].initial.print_manager.last_x'
          : 'players[' + index + '].initial.print_manager.lastX'
      );
    }
    lastY = firstDefined(printManager, ['last_y', 'lastY']);
    if (typeof lastY !== 'undefined') {
      avatar.printManager.lastY = requiredNumber(
        lastY,
        ownProperty(printManager, 'last_y')
          ? 'players[' + index + '].initial.print_manager.last_y'
          : 'players[' + index + '].initial.print_manager.lastY'
      );
    }
  }

  function requiredTrailPoint(value, field) {
    if (!Array.isArray(value) || value.length !== 2) {
      throw new Error(field + ' must be a two-number point');
    }
    return [
      requiredNumber(value[0], field + '[0]'),
      requiredNumber(value[1], field + '[1]')
    ];
  }

  function applyForcedTrailState(avatar, playerSpec, index) {
    var state = stateFor(playerSpec);
    var trail = firstDefined(state, ['trail']);
    var inferredLastPoint = null;
    var lastX;
    var lastY;

    if (typeof trail === 'undefined') {
      return;
    }
    if (!trail || typeof trail !== 'object') {
      throw new Error('players[' + index + '].initial.trail must be an object');
    }

    if (ownProperty(trail, 'points')) {
      if (!Array.isArray(trail.points)) {
        throw new Error('players[' + index + '].initial.trail.points must be an array');
      }
      avatar.trail.points.length = 0;
      for (var pointIndex = 0; pointIndex < trail.points.length; pointIndex++) {
        avatar.trail.points.push(requiredTrailPoint(
          trail.points[pointIndex],
          'players[' + index + '].initial.trail.points[' + pointIndex + ']'
        ));
      }
      inferredLastPoint = avatar.trail.points.length
        ? avatar.trail.points[avatar.trail.points.length - 1]
        : null;
    }
    lastX = firstDefined(trail, ['last_x', 'lastX']);
    if (typeof lastX !== 'undefined') {
      avatar.trail.lastX = lastX === null
        ? null
        : requiredNumber(
          lastX,
          ownProperty(trail, 'last_x')
            ? 'players[' + index + '].initial.trail.last_x'
            : 'players[' + index + '].initial.trail.lastX'
        );
    } else if (ownProperty(trail, 'points')) {
      avatar.trail.lastX = inferredLastPoint ? inferredLastPoint[0] : null;
    }
    lastY = firstDefined(trail, ['last_y', 'lastY']);
    if (typeof lastY !== 'undefined') {
      avatar.trail.lastY = lastY === null
        ? null
        : requiredNumber(
          lastY,
          ownProperty(trail, 'last_y')
            ? 'players[' + index + '].initial.trail.last_y'
            : 'players[' + index + '].initial.trail.lastY'
        );
    } else if (ownProperty(trail, 'points')) {
      avatar.trail.lastY = inferredLastPoint ? inferredLastPoint[1] : null;
    }
  }

  function applyForcedBodyState(avatar, playerSpec, index) {
    var bodyCount = optionalInteger(
      forcedStateValue(playerSpec, ['body_count', 'bodyCount']),
      'players[' + index + '].initial.body_count'
    );
    var bodyNum = optionalInteger(
      forcedStateValue(playerSpec, ['body_num', 'bodyNum']),
      'players[' + index + '].initial.body_num'
    );

    if (bodyCount !== null) {
      avatar.bodyCount = bodyCount;
      avatar.body.num = bodyCount;
    }
    if (bodyNum !== null) {
      avatar.body.num = bodyNum;
    }
  }

  function moveFor(tick, avatar, index) {
    var moves = tick.moves;
    var spec = playerSpecs[index] || {};

    if (Array.isArray(moves)) {
      if (typeof moves[index] === 'number') {
        return moves[index];
      }

      for (var i = 0; i < moves.length; i++) {
        var item = moves[i] || {};
        if (
          item.player_id === spec.id ||
          item.playerId === spec.id ||
          item.avatar_id === avatar.id ||
          item.avatarId === avatar.id ||
          item.avatar === avatar.id ||
          item.name === avatar.name
        ) {
          return numberOr(item.move, 0);
        }
      }

      return 0;
    }

    if (moves && typeof moves === 'object') {
      if (typeof moves[spec.id] === 'number') {
        return moves[spec.id];
      }
      if (typeof moves[avatar.id] === 'number') {
        return moves[avatar.id];
      }
      if (typeof moves[avatar.name] === 'number') {
        return moves[avatar.name];
      }
      if (typeof moves[index] === 'number') {
        return moves[index];
      }
    }

    if (typeof tick.move === 'number') {
      return tick.move;
    }

    return 0;
  }

  function maxScore() {
    var roomConfig = scenario.source_setup && scenario.source_setup.room;
    return numberOr(scenario.max_score, numberOr(scenario.maxScore, roomConfig ? numberOr(roomConfig.max_score, 10) : 10));
  }

  function bonusConstructorFor(type, field) {
    var constructors = {
      BonusSelfSmall: BonusSelfSmall,
      BonusSelfSlow: BonusSelfSlow,
      BonusSelfFast: BonusSelfFast,
      BonusSelfMaster: BonusSelfMaster,
      BonusEnemySlow: BonusEnemySlow,
      BonusEnemyFast: BonusEnemyFast,
      BonusEnemyBig: BonusEnemyBig,
      BonusEnemyInverse: BonusEnemyInverse,
      BonusEnemyStraightAngle: BonusEnemyStraightAngle,
      BonusGameBorderless: BonusGameBorderless,
      BonusAllColor: BonusAllColor,
      BonusGameClear: BonusGameClear
    };
    if (!constructors[type]) {
      throw new Error(
        field + ' must be one of: ' + Object.keys(constructors).join(', ')
      );
    }
    return constructors[type];
  }

  function roomConfig() {
    return scenario.source_setup && scenario.source_setup.room &&
      typeof scenario.source_setup.room === 'object'
      ? scenario.source_setup.room
      : {};
  }

  function roomBonusTypes() {
    var config = roomConfig();
    var bonuses = firstDefined(config, ['bonuses', 'bonus_types', 'bonusTypes']);
    var types = [];

    if (typeof bonuses === 'undefined') {
      return types;
    }
    if (!Array.isArray(bonuses)) {
      throw new Error('source_setup.room.bonuses must be an array');
    }

    for (var index = 0; index < bonuses.length; index++) {
      types.push(bonusConstructorFor(
        requiredString(bonuses[index], 'source_setup.room.bonuses[' + index + ']'),
        'source_setup.room.bonuses[' + index + ']'
      ));
    }
    return types;
  }

  function bonusRate() {
    var config = roomConfig();
    var value = firstDefined(config, ['bonus_rate', 'bonusRate']);
    return typeof value === 'undefined'
      ? 0
      : requiredNumber(value, ownProperty(config, 'bonus_rate')
        ? 'source_setup.room.bonus_rate'
        : 'source_setup.room.bonusRate');
  }

  function timePolicy() {
    return scenario.time_policy && typeof scenario.time_policy === 'object'
      ? scenario.time_policy
      : {};
  }

  function timerAdvanceFor(tick, index) {
    var direct = firstDefined(tick, [
      'advance_timers_ms',
      'advanceTimersMs',
      'timer_advance_ms',
      'timerAdvanceMs'
    ]);
    var sequence;

    if (typeof direct !== 'undefined') {
      return requiredNumber(direct, 'ticks[' + index + '].advance_timers_ms');
    }

    sequence = firstDefined(timePolicy(), [
      'timer_advance_ms_sequence',
      'timerAdvanceMsSequence'
    ]);
    if (typeof sequence !== 'undefined') {
      if (!Array.isArray(sequence)) {
        throw new Error('time_policy.timer_advance_ms_sequence must be an array');
      }
      if (index >= sequence.length) {
        return 0;
      }
      return requiredNumber(
        sequence[index],
        'time_policy.timer_advance_ms_sequence[' + index + ']'
      );
    }

    return 0;
  }

  function scenarioId() {
    return scenario.scenario_id || scenario.id || 'unnamed';
  }

  function initialState() {
    return scenario.initial_state && typeof scenario.initial_state === 'object'
      ? scenario.initial_state
      : {};
  }

  function makeRoom(playerSpecs) {
    var config = roomConfig();
    return {
      name: config.name || scenarioId() || 'oracle-scenario',
      players: new Collection([], 'id', true),
      config: {
        getMaxScore: function () { return maxScore(); },
        getBonuses: function () { return roomBonusTypes(); },
        getVariable: function (name) { return name === 'bonusRate' ? bonusRate() : undefined; }
      },
      controller: { clients: new Collection() }
    };
  }

  function ownerIdFor(bodySpec) {
    return bodySpec.player_id ||
      bodySpec.playerId ||
      bodySpec.owner_id ||
      bodySpec.ownerId ||
      bodySpec.avatar_id ||
      bodySpec.avatarId ||
      bodySpec.avatar;
  }

  function valuesEqual(left, right) {
    return typeof left !== 'undefined' &&
      typeof right !== 'undefined' &&
      String(left) === String(right);
  }

  function findAvatarByOwner(avatars, playerSpecs, ownerId) {
    for (var index = 0; index < avatars.length; index++) {
      var avatar = avatars[index];
      var spec = playerSpecs[index] || {};
      if (
        valuesEqual(ownerId, spec.id) ||
        valuesEqual(ownerId, spec.player_id) ||
        valuesEqual(ownerId, spec.playerId) ||
        valuesEqual(ownerId, spec.avatar_id) ||
        valuesEqual(ownerId, spec.avatarId) ||
        valuesEqual(ownerId, spec.name) ||
        valuesEqual(ownerId, avatar.id) ||
        valuesEqual(ownerId, avatar.name)
      ) {
        return avatar;
      }
    }
    return null;
  }

  function worldBodySpecs() {
    var state = initialState();
    return state.world_bodies || state.worldBodies || [];
  }

  function activeBonusSpecs() {
    var state = initialState();
    return state.active_bonuses || state.activeBonuses || [];
  }

  function seedActiveBonusesAfterOnStart() {
    var state = initialState();
    var gameSetup = scenario.source_setup && scenario.source_setup.game
      ? scenario.source_setup.game
      : {};
    return state.seed_active_bonuses_after_on_start === true ||
      state.seedActiveBonusesAfterOnStart === true ||
      gameSetup.seed_active_bonuses_after_on_start === true ||
      gameSetup.seedActiveBonusesAfterOnStart === true;
  }

  function seedWorldBodies(game, avatars, playerSpecs) {
    var specs = worldBodySpecs();
    if (!Array.isArray(specs)) {
      throw new Error('initial_state.world_bodies must be an array');
    }
    if (specs.length > 0 && !game.world.active) {
      throw new Error('initial_state.world_bodies requires source_setup.game.world_active not to be false');
    }

    for (var index = 0; index < specs.length; index++) {
      var spec = specs[index] || {};
      if (typeof spec !== 'object') {
        throw new Error('initial_state.world_bodies[' + index + '] must be an object');
      }
      var ownerId = ownerIdFor(spec);
      if (typeof ownerId === 'undefined' || ownerId === null || ownerId === '') {
        throw new Error('initial_state.world_bodies[' + index + '].player_id is required');
      }
      var avatar = findAvatarByOwner(avatars, playerSpecs, ownerId);
      if (!avatar) {
        throw new Error('initial_state.world_bodies[' + index + '] references unknown player ' + ownerId);
      }

      var body = new AvatarBody(
        requiredNumber(spec.x, 'initial_state.world_bodies[' + index + '].x'),
        requiredNumber(spec.y, 'initial_state.world_bodies[' + index + '].y'),
        avatar
      );
      var ageMs = optionalNonNegativeNumber(
        firstDefined(spec, ['age_ms', 'ageMs']),
        'initial_state.world_bodies[' + index + '].age_ms'
      );
      body.radius = requiredNumber(spec.radius, 'initial_state.world_bodies[' + index + '].radius');
      body.num = requiredInteger(spec.num, 'initial_state.world_bodies[' + index + '].num');
      if (ageMs !== null) {
        body.birth = new Date().getTime() - ageMs;
      }
      avatar.bodyCount = Math.max(avatar.bodyCount, body.num + 1);
      game.world.addBody(body);
    }
  }

  function seedActiveBonuses(game) {
    var specs = activeBonusSpecs();
    if (!Array.isArray(specs)) {
      throw new Error('initial_state.active_bonuses must be an array');
    }
    if (!specs.length) {
      return;
    }

    game.bonusManager.world.activate();

    for (var index = 0; index < specs.length; index++) {
      var spec = specs[index] || {};
      if (typeof spec !== 'object') {
        throw new Error('initial_state.active_bonuses[' + index + '] must be an object');
      }
      var type = requiredString(
        firstDefined(spec, ['type', 'bonus_type', 'bonusType']),
        'initial_state.active_bonuses[' + index + '].type'
      );
      var BonusType = bonusConstructorFor(
        type,
        'initial_state.active_bonuses[' + index + '].type'
      );
      var bonus = new BonusType(
        requiredNumber(spec.x, 'initial_state.active_bonuses[' + index + '].x'),
        requiredNumber(spec.y, 'initial_state.active_bonuses[' + index + '].y')
      );
      if (!game.bonusManager.add(bonus)) {
        throw new Error('initial_state.active_bonuses[' + index + '] could not be seeded');
      }
    }
  }

  var playerSpecs = scenario.players || initialState().players || [];
  var room = makeRoom(playerSpecs);

  for (var i = 0; i < playerSpecs.length; i++) {
    var spec = playerSpecs[i];
    var name = spec.name || spec.id || 'p' + i;
    var player = new Player({ id: name + '-client', active: true }, name, spec.color || '#ffffff');

    if (typeof spec.avatar_id !== 'undefined') {
      player.id = spec.avatar_id;
    } else if (typeof spec.avatarId !== 'undefined') {
      player.id = spec.avatarId;
    } else if (typeof spec.id !== 'undefined') {
      player.id = spec.id;
    }

    room.players.add(player);
  }

  var game = new Game(room);
  var avatars = game.avatars.items;

  [
    'round:new',
    'round:end',
    'clear',
    'borderless',
    'game:start',
    'game:stop',
    'end'
  ].forEach(function (name) { record(game, name); });

  ['bonus:pop', 'bonus:clear'].forEach(function (name) {
    record(game.bonusManager, name);
  });

  avatars.forEach(function (avatar) {
    [
      'position',
      'angle',
      'point',
      'die',
      'score',
      'score:round',
      'property'
    ].forEach(function (name) { record(avatar, name); });

    record(avatar.bonusStack, 'change', 'bonus:stack');
  });

  var gameSetup = scenario.source_setup && scenario.source_setup.game ? scenario.source_setup.game : {};
  game.started = typeof gameSetup.started === 'boolean' ? gameSetup.started : true;
  game.inRound = typeof gameSetup.in_round === 'boolean' ? gameSetup.in_round : true;
  game.borderless = typeof gameSetup.borderless === 'boolean' ? gameSetup.borderless : game.borderless;

  if (gameSetup.world_active === false) {
    game.world.clear();
  } else {
    game.world.activate();
  }

  for (i = 0; i < avatars.length; i++) {
    applyForcedState(avatars[i], playerSpecs[i] || {});
  }

  for (i = 0; i < avatars.length; i++) {
    applyForcedTrailState(avatars[i], playerSpecs[i] || {}, i);
    applyForcedPrintManagerState(avatars[i], playerSpecs[i] || {}, i);
  }

  seedWorldBodies(game, avatars, playerSpecs);
  if (!seedActiveBonusesAfterOnStart()) {
    seedActiveBonuses(game);
  }

  for (i = 0; i < avatars.length; i++) {
    applyForcedBodyState(avatars[i], playerSpecs[i] || {}, i);
  }

  if (gameSetup.invoke_on_start === true || gameSetup.invokeOnStart === true) {
    game.onStart();
  }

  if (seedActiveBonusesAfterOnStart()) {
    seedActiveBonuses(game);
  }

  events.length = 0;

  var ticks = scenario.ticks || scenario.steps || [{ moves: scenario.moves || {} }];
  var trace = [];

  for (i = 0; i < ticks.length; i++) {
    var tick = ticks[i] || {};
    var movesUsed = [];
    var stepMs = numberOr(tick.step_ms, numberOr(tick.stepMs, numberOr(scenario.step_ms, numberOr(scenario.stepMs, 1000 / 60))));

    events.length = 0;
    __advanceTimers(timerAdvanceFor(tick, i));

    for (var j = 0; j < avatars.length; j++) {
      var move = moveFor(tick, avatars[j], j);
      movesUsed[j] = move;
      avatars[j].updateAngularVelocity(move);
    }

    game.update(stepMs);

    trace.push({
      scenario: scenarioId(),
      playerCount: avatars.length,
      tick: i,
      stepMs: round(stepMs),
      game: snapshotGame(game),
      avatars: avatars.map(function (avatar, index) {
        return snapshotAvatar(avatar, movesUsed[index]);
      }),
      events: events.slice()
    });
  }

  return {
    scenario: scenarioId(),
    playerCount: avatars.length,
    comparison: scenario.comparison || {},
    trace: trace
  };
}(__scenario))
`, context, { filename: 'scenario_runner.vm.js' });
}

function main() {
  const { source, scenario } = readScenario();
  const context = makeContext(scenario);

  loadOriginalSources(context);

  const result = runScenario(context, scenario);
  result.source = source;
  result.loadedSources = sourceFiles;
  result.randomCalls = context.__randomCalls.slice();

  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`${JSON.stringify({
    error: {
      name: error.name,
      message: error.message,
      stack: error.stack,
    },
  }, null, 2)}\n`);
  process.exitCode = 1;
}
