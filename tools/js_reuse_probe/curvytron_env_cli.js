#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { EventEmitter } = require('events');

const repoRoot = path.resolve(__dirname, '..', '..');
const referenceRoot = path.join(repoRoot, 'third_party', 'curvytron-reference');
const defaultScenarioPath = path.join(
  repoRoot,
  'scenarios',
  'environment',
  'source_kinematics_turn_multistep.json'
);

const sourceFiles = [
  'src/shared/Collection.js',
  'src/shared/service/BaseFPSLogger.js',
  'src/server/service/FPSLogger.js',
  'src/shared/service/Compressor.js',
  'src/shared/model/BaseBonus.js',
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

function requiredNumber(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${field} must be a finite number`);
  }
  return value;
}

function optionalNonNegativeNumber(value, field) {
  if (typeof value === 'undefined' || value === null) {
    return null;
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new Error(`${field} must be a non-negative finite number`);
  }
  return value;
}

function requiredInteger(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.floor(value) !== value) {
    throw new Error(`${field} must be an integer`);
  }
  return value;
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
    return {
      sequence: rawSequence.map((value, index) => requiredRandomValue(
        value,
        `source_setup.random.math_random_sequence[${index}]`
      )),
      constant: null,
    };
  }

  return {
    sequence: null,
    constant: requiredRandomValue(
      firstOwned(random, ['math_random', 'mathRandom']) ?? 0.5,
      'source_setup.random.math_random'
    ),
  };
}

function makeControlledDate(clock) {
  const RealDate = Date;

  function ControlledDate(...args) {
    if (this instanceof ControlledDate) {
      if (args.length > 0) {
        return new RealDate(...args);
      }
      return new RealDate(clock.nowMs);
    }
    return args.length > 0 ? RealDate(...args) : RealDate(clock.nowMs);
  }

  ControlledDate.prototype = RealDate.prototype;
  ControlledDate.now = function now() {
    return clock.nowMs;
  };
  ControlledDate.parse = RealDate.parse;
  ControlledDate.UTC = RealDate.UTC;

  return ControlledDate;
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
    randomCalls.push({ index, value });
    index += 1;
    return value;
  };

  return math;
}

function makeContext(scenario) {
  const timers = [];
  const randomCalls = [];
  const clock = { nowMs: 0 };
  let nextTimerId = 1;
  let nextTimerOrder = 1;

  function normalizedDelay(delay) {
    const value = Number(delay);
    return Number.isFinite(value) && value > 0 ? value : 0;
  }

  function scheduleTimer(callback, delay, interval) {
    const normalized = normalizedDelay(delay);
    const timer = {
      type: interval ? 'interval' : 'timeout',
      id: nextTimerId++,
      delay: normalized,
      dueAt: clock.nowMs + normalized,
      order: nextTimerOrder++,
      callback,
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

    const targetMs = clock.nowMs + advanceMs;
    let guard = 0;
    let timer = nextDueTimer(targetMs);
    while (timer) {
      if (guard++ > 100000) {
        throw new Error('timer advance exceeded callback guard');
      }
      clock.nowMs = timer.dueAt;
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
    clock.nowMs = targetMs;
    return clock.nowMs;
  }

  return vm.createContext({
    console,
    EventEmitter,
    Math: makeDeterministicMath(scenario, randomCalls),
    Date: makeControlledDate(clock),
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
    __nowMs() {
      return clock.nowMs;
    },
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

function readScenario() {
  const requested = process.argv[2];
  const fixture = requested ? path.resolve(requested) : defaultScenarioPath;

  if (!fs.existsSync(fixture)) {
    throw new Error(`Scenario file not found: ${fixture}`);
  }

  return {
    source: relative(fixture),
    scenario: JSON.parse(fs.readFileSync(fixture, 'utf8')),
  };
}

function runEnvProbe(context, scenario) {
  context.__scenario = scenario;

  return vm.runInContext(`
(function (scenario) {
  var events = [];
  var previousScores = [];
  var previousRoundScores = [];

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

  function requiredInteger(value, field) {
    if (typeof value !== 'number' || !isFinite(value) || Math.floor(value) !== value) {
      throw new Error(field + ' must be an integer');
    }
    return value;
  }

  function optionalInteger(value, field) {
    return typeof value === 'undefined' || value === null ? null : requiredInteger(value, field);
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

  function scenarioId() {
    return scenario.scenario_id || scenario.id || 'unnamed';
  }

  function sourceSetup() {
    return scenario.source_setup && typeof scenario.source_setup === 'object'
      ? scenario.source_setup
      : {};
  }

  function initialState() {
    return scenario.initial_state && typeof scenario.initial_state === 'object'
      ? scenario.initial_state
      : {};
  }

  function playerSpecs() {
    return scenario.players || initialState().players || playersFromLegacyInitialState();
  }

  function playerCount() {
    return numberOr(
      sourceSetup().player_count,
      numberOr(scenario.player_count, numberOr(initialState().player_count, playerSpecs().length))
    );
  }

  function playersFromLegacyInitialState() {
    var count = numberOr(scenario.player_count, numberOr(sourceSetup().player_count, 0));
    var positions = initialState().positions || [];
    var headings = initialState().headings || [];
    var players = [];
    for (var i = 0; i < count; i++) {
      players.push({
        id: 'p' + i,
        avatar_id: i + 1,
        name: 'p' + i,
        initial: {
          x: positions[i] ? positions[i][0] : undefined,
          y: positions[i] ? positions[i][1] : undefined,
          angle_rad: typeof headings[i] === 'number' ? headings[i] : 0,
          printing: false
        }
      });
    }
    return players;
  }

  function mapSizeForFallback(count) {
    var mapSize = firstDefined(sourceSetup(), ['map_size', 'mapSize']);
    if (typeof mapSize !== 'undefined') {
      return requiredNumber(mapSize, 'source_setup.map_size');
    }
    var initialMapSize = firstDefined(initialState(), ['map_size', 'mapSize']);
    if (typeof initialMapSize !== 'undefined') {
      return requiredNumber(initialMapSize, 'initial_state.map_size');
    }
    if (count <= 0) {
      return 80;
    }
    return Math.round(Math.sqrt(80 * 80 + ((count - 1) * 80 * 80 / 5)));
  }

  function maxScore() {
    var roomConfig = sourceSetup().room || {};
    return numberOr(scenario.max_score, numberOr(scenario.maxScore, numberOr(roomConfig.max_score, 10)));
  }

  function roomConfigVariable(name) {
    var roomConfig = sourceSetup().room || {};
    if (name === 'bonusRate') {
      return numberOr(roomConfig.bonus_rate, numberOr(roomConfig.bonusRate, 0));
    }
    return undefined;
  }

  function makeRoom(specs) {
    return {
      name: (sourceSetup().room && sourceSetup().room.name) || scenarioId(),
      players: new Collection([], 'id', true),
      config: {
        getMaxScore: function () { return maxScore(); },
        getBonuses: function () { return []; },
        getVariable: roomConfigVariable
      },
      controller: { clients: new Collection() }
    };
  }

  function addPlayers(room, specs) {
    for (var i = 0; i < specs.length; i++) {
      var spec = specs[i] || {};
      var name = spec.name || spec.id || 'p' + i;
      var player = new Player(
        { id: (spec.client_id || spec.clientId || name + '-client'), active: true },
        name,
        spec.color || '#ffffff'
      );

      if (typeof spec.avatar_id !== 'undefined') {
        player.id = spec.avatar_id;
      } else if (typeof spec.avatarId !== 'undefined') {
        player.id = spec.avatarId;
      } else if (typeof spec.id !== 'undefined') {
        player.id = spec.id;
      }

      room.players.add(player);
    }
  }

  function eventData(name, data) {
    switch (name) {
      case 'position':
        return { player: data.id, x: round(data.x), y: round(data.y) };
      case 'angle':
        return { player: data.id, angle: round(data.angle) };
      case 'point':
        return {
          player: data.avatar.id,
          x: round(data.x),
          y: round(data.y),
          important: data.important ? true : false
        };
      case 'die':
        return {
          player: data.avatar.id,
          killer: data.killer ? data.killer.id : null,
          old: typeof data.old === 'undefined' ? null : data.old
        };
      case 'score':
      case 'score:round':
        return { player: data.id, score: data.score, roundScore: data.roundScore };
      case 'property':
        return { player: data.avatar.id, property: data.property, value: data.value };
      case 'round:end':
        return { winner: data.winner ? data.winner.id : null };
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

  function stateFor(playerSpec) {
    return playerSpec.state || playerSpec.initial || playerSpec;
  }

  function stateAngle(state, fallback) {
    return numberOr(state.angle, numberOr(state.angle_rad, fallback));
  }

  function hasForcedPrintRuntimeState(state) {
    return state && typeof state === 'object' && (
      ownProperty(state, 'print_manager') ||
      ownProperty(state, 'printManager') ||
      ownProperty(state, 'trail')
    );
  }

  function applyForcedState(avatar, spec) {
    var state = stateFor(spec || {});

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
    } else if (state.printing === true) {
      avatar.printManager.start();
    }
  }

  function applyForcedPrintManagerState(avatar, spec, index) {
    var state = stateFor(spec || {});
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
      avatar.printManager.lastX = requiredNumber(lastX, 'players[' + index + '].initial.print_manager.last_x');
    }
    lastY = firstDefined(printManager, ['last_y', 'lastY']);
    if (typeof lastY !== 'undefined') {
      avatar.printManager.lastY = requiredNumber(lastY, 'players[' + index + '].initial.print_manager.last_y');
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

  function applyForcedTrailState(avatar, spec, index) {
    var state = stateFor(spec || {});
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
      avatar.trail.lastX = lastX === null ? null : requiredNumber(lastX, 'players[' + index + '].initial.trail.last_x');
    } else if (ownProperty(trail, 'points')) {
      avatar.trail.lastX = inferredLastPoint ? inferredLastPoint[0] : null;
    }

    lastY = firstDefined(trail, ['last_y', 'lastY']);
    if (typeof lastY !== 'undefined') {
      avatar.trail.lastY = lastY === null ? null : requiredNumber(lastY, 'players[' + index + '].initial.trail.last_y');
    } else if (ownProperty(trail, 'points')) {
      avatar.trail.lastY = inferredLastPoint ? inferredLastPoint[1] : null;
    }
  }

  function forcedStateValue(spec, keys) {
    var state = stateFor(spec || {});
    var value = firstDefined(state, keys);
    return typeof value === 'undefined' ? firstDefined(spec, keys) : value;
  }

  function applyForcedBodyState(avatar, spec, index) {
    var bodyCount = optionalInteger(
      forcedStateValue(spec, ['body_count', 'bodyCount']),
      'players[' + index + '].initial.body_count'
    );
    var bodyNum = optionalInteger(
      forcedStateValue(spec, ['body_num', 'bodyNum']),
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

  function findAvatarByOwner(avatars, specs, ownerId) {
    for (var index = 0; index < avatars.length; index++) {
      var avatar = avatars[index];
      var spec = specs[index] || {};
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
    return initialState().world_bodies || initialState().worldBodies || [];
  }

  function seedWorldBodies(game, avatars, specs) {
    var bodies = worldBodySpecs();
    if (!Array.isArray(bodies)) {
      throw new Error('initial_state.world_bodies must be an array');
    }
    if (bodies.length > 0 && !game.world.active) {
      throw new Error('initial_state.world_bodies requires source_setup.game.world_active not to be false');
    }

    for (var index = 0; index < bodies.length; index++) {
      var spec = bodies[index] || {};
      var ownerId = ownerIdFor(spec);
      if (typeof ownerId === 'undefined' || ownerId === null || ownerId === '') {
        throw new Error('initial_state.world_bodies[' + index + '].player_id is required');
      }

      var avatar = findAvatarByOwner(avatars, specs, ownerId);
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

  function resetMode() {
    var configured = scenario.env_reset && scenario.env_reset.mode;
    if (configured) {
      return configured;
    }
    if (scenario.lifecycle && typeof scenario.lifecycle === 'object') {
      return 'source_new_round';
    }
    return 'forced_state';
  }

  function configureForcedReset(game, avatars, specs) {
    var gameSetup = sourceSetup().game || {};
    game.started = typeof gameSetup.started === 'boolean' ? gameSetup.started : true;
    game.inRound = typeof gameSetup.in_round === 'boolean' ? gameSetup.in_round : true;
    game.borderless = typeof gameSetup.borderless === 'boolean' ? gameSetup.borderless : game.borderless;

    if (typeof gameSetup.map_size === 'number') {
      game.size = gameSetup.map_size;
      game.world.clear();
      game.world = new World(game.size);
    } else {
      var size = mapSizeForFallback(avatars.length);
      if (game.size !== size) {
        game.size = size;
        game.world.clear();
        game.world = new World(game.size);
      }
    }

    if (gameSetup.world_active === false) {
      game.world.clear();
    } else {
      game.world.activate();
    }

    for (var i = 0; i < avatars.length; i++) {
      applyForcedState(avatars[i], specs[i] || {});
    }
    for (i = 0; i < avatars.length; i++) {
      applyForcedTrailState(avatars[i], specs[i] || {}, i);
      applyForcedPrintManagerState(avatars[i], specs[i] || {}, i);
    }
    seedWorldBodies(game, avatars, specs);
    for (i = 0; i < avatars.length; i++) {
      applyForcedBodyState(avatars[i], specs[i] || {}, i);
    }
  }

  function configureNaturalReset(game, avatars, specs) {
    var lifecycle = scenario.lifecycle || {};
    var warmupMs = numberOr(lifecycle.new_round_time_ms, numberOr(scenario.new_round_time_ms, 0));
    for (var i = 0; i < avatars.length && i < specs.length; i++) {
      var state = stateFor(specs[i] || {});
      if (typeof state.present === 'boolean') {
        avatars[i].present = state.present;
      }
    }
    game.started = false;
    game.inRound = false;
    game.newRound(warmupMs);
  }

  function snapshotAvatar(avatar) {
    return {
      id: avatar.id,
      name: avatar.name,
      x: round(avatar.x),
      y: round(avatar.y),
      angle: round(avatar.angle),
      alive: avatar.alive,
      present: avatar.present,
      printing: avatar.printing,
      score: avatar.score,
      roundScore: avatar.roundScore,
      trailPoints: avatar.trail.points.length,
      bodyNum: avatar.body ? avatar.body.num : null,
      bodyCount: avatar.bodyCount,
      print: {
        active: avatar.printManager.active,
        distance: round(avatar.printManager.distance),
        lastX: round(avatar.printManager.lastX),
        lastY: round(avatar.printManager.lastY)
      }
    };
  }

  function snapshotState(game, avatars) {
    var hasWorld = game.world && typeof game.world === 'object';
    return {
      tMs: round(__nowMs()),
      game: {
        size: game.size,
        started: game.started,
        inRound: game.inRound,
        borderless: game.borderless,
        worldActive: hasWorld ? game.world.active : null,
        worldBodyCount: hasWorld ? game.world.bodyCount : null,
        deaths: game.deaths && game.deaths.items ? game.deaths.items.map(function (avatar) { return avatar.id; }) : [],
        roundWinner: game.roundWinner ? game.roundWinner.id : null,
        gameWinner: game.gameWinner ? game.gameWinner.id : null
      },
      players: avatars.map(snapshotAvatar)
    };
  }

  function rememberScores(avatars) {
    previousScores = avatars.map(function (avatar) { return avatar.score; });
    previousRoundScores = avatars.map(function (avatar) { return avatar.roundScore; });
  }

  function rewards(avatars) {
    var out = [];
    for (var i = 0; i < avatars.length; i++) {
      out.push(round(
        avatars[i].score - previousScores[i] +
        avatars[i].roundScore - previousRoundScores[i]
      ));
    }
    return out;
  }

  function moveFor(step, avatar, index, specs) {
    var moves = step.moves;
    var spec = specs[index] || {};

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

    if (typeof step.move === 'number') {
      return step.move;
    }

    return 0;
  }

  function steps() {
    var raw = scenario.steps || scenario.ticks || scenario.action_script || [];
    if (!Array.isArray(raw)) {
      throw new Error('scenario steps must be an array');
    }
    return raw;
  }

  function timePolicy() {
    return scenario.time_policy && typeof scenario.time_policy === 'object'
      ? scenario.time_policy
      : {};
  }

  function stepMsFor(step, index) {
    return numberOr(
      step.step_ms,
      numberOr(step.stepMs, numberOr(timePolicy().step_ms, numberOr(timePolicy().stepMs, 1000 / 60)))
    );
  }

  function timerAdvanceFor(step, index) {
    var direct = firstDefined(step, [
      'advance_timers_ms',
      'advanceTimersMs',
      'timer_advance_ms',
      'timerAdvanceMs'
    ]);
    var sequence;

    if (typeof direct !== 'undefined') {
      return requiredNumber(direct, 'steps[' + index + '].advance_timers_ms');
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
      return requiredNumber(sequence[index], 'time_policy.timer_advance_ms_sequence[' + index + ']');
    }

    return 0;
  }

  var specs = playerSpecs();
  if (!Array.isArray(specs)) {
    throw new Error('players must be an array');
  }

  var expectedCount = playerCount();
  if (specs.length !== expectedCount) {
    throw new Error('player count mismatch: expected ' + expectedCount + ', got ' + specs.length);
  }

  var room = makeRoom(specs);
  addPlayers(room, specs);

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
  });

  var mode = resetMode();
  if (mode === 'source_new_round') {
    configureNaturalReset(game, avatars, specs);
  } else if (mode === 'forced_state') {
    configureForcedReset(game, avatars, specs);
  } else {
    throw new Error('unsupported env_reset.mode: ' + mode);
  }

  var resetEvents = events.slice();
  events.length = 0;
  rememberScores(avatars);

  var reset = {
    mode: mode,
    state: snapshotState(game, avatars),
    events: resetEvents
  };

  var rawSteps = steps();
  var frames = [];
  for (var i = 0; i < rawSteps.length; i++) {
    var step = rawSteps[i] || {};
    var stepMs = requiredNumber(stepMsFor(step, i), 'steps[' + i + '].step_ms');
    var timerAdvanceMs = timerAdvanceFor(step, i);
    var movesUsed = [];
    var frameEvents;

    events.length = 0;
    __advanceTimers(timerAdvanceMs);

    for (var j = 0; j < avatars.length; j++) {
      var move = moveFor(step, avatars[j], j, specs);
      movesUsed.push({ player: avatars[j].id, move: move });
      avatars[j].updateAngularVelocity(move);
    }

    game.update(stepMs);
    frameEvents = events.slice();

    frames.push({
      tick: typeof step.tick === 'number' ? step.tick : i,
      stepMs: round(stepMs),
      timerAdvanceMs: round(timerAdvanceMs),
      moves: movesUsed,
      reward: rewards(avatars),
      roundDone: !game.inRound,
      gameDone: !game.started || !game.avatars,
      state: snapshotState(game, avatars),
      events: frameEvents
    });
    rememberScores(avatars);
  }

  return {
    schema: 'curvytron-js-reuse-env-probe-v0',
    runner: 'original-curvytron-js-vm-batch-cli',
    scenario: scenarioId(),
    capabilities: {
      originalSourceLoaded: true,
      deterministicMathRandom: true,
      manualTimerAdvance: true,
      resetCall: mode,
      stepCall: 'avatar.updateAngularVelocity(move) + game.update(stepMs)'
    },
    reset: reset,
    steps: frames,
    randomCalls: __randomCalls.slice()
  };
}(__scenario))
`, context, { filename: 'curvytron_env_probe.vm.js' });
}

function main() {
  const { source, scenario } = readScenario();
  const context = makeContext(scenario);

  loadOriginalSources(context);

  const result = runEnvProbe(context, scenario);
  result.source = source;
  result.sourceRoot = relative(referenceRoot);
  result.loadedSourceCount = sourceFiles.length;

  process.stdout.write(`${JSON.stringify(result)}\n`);
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
