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
  'source_lifecycle_spawn_rng_warmup_print_start_2p.json'
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

function requiredRandomValue(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value >= 1) {
    throw new Error(`${field} must be a finite number in [0, 1)`);
  }
  return value;
}

function randomTape(scenario) {
  const sourceSetup = scenario && scenario.source_setup;
  const random = sourceSetup && sourceSetup.random;
  const rawSequence = firstOwned(random, [
    'math_random_sequence',
    'mathRandomSequence',
    'math_random_tape',
    'mathRandomTape',
  ]);

  if (!Array.isArray(rawSequence)) {
    throw new Error('source_setup.random.math_random_sequence must be an array');
  }

  return rawSequence.map((value, index) => requiredRandomValue(
    value,
    `source_setup.random.math_random_sequence[${index}]`
  ));
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

function makeDeterministicMath(scenario, randomState) {
  const sequence = randomTape(scenario);
  const math = Object.create(Math);
  let index = 0;

  math.random = function random() {
    if (index >= sequence.length) {
      throw new Error(`Math.random tape exhausted after ${index} calls`);
    }

    const value = sequence[index];
    const label = randomState.labels.length
      ? randomState.labels[randomState.labels.length - 1]
      : { site: 'unlabeled' };
    const call = {
      index,
      value,
      atMs: randomState.clock.nowMs,
      label: Object.assign({}, label),
    };

    randomState.calls.push(call);
    if (typeof randomState.observer === 'function') {
      randomState.observer(call);
    }
    index += 1;
    return value;
  };

  return math;
}

function makeContext(scenario) {
  const timers = [];
  const clock = { nowMs: 0 };
  const randomState = {
    calls: [],
    labels: [],
    observer: null,
    clock,
  };
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
    const normalized = normalizedDelay(delay);
    const timer = {
      type: interval ? 'interval' : 'timeout',
      id: nextTimerId++,
      delay: normalized,
      dueAt: clock.nowMs + normalized,
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
    Math: makeDeterministicMath(scenario, randomState),
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
    __pushRandomLabel(label) {
      randomState.labels.push(label);
    },
    __popRandomLabel() {
      randomState.labels.pop();
    },
    __setRandomObserver(observer) {
      randomState.observer = observer;
    },
    __randomCalls: randomState.calls,
    __timers: timers,
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
  const fixture = requested ? path.resolve(requested) : defaultFixturePath;

  if (!fs.existsSync(fixture)) {
    throw new Error(`Scenario file not found: ${fixture}`);
  }

  return {
    source: relative(fixture),
    scenario: JSON.parse(fs.readFileSync(fixture, 'utf8')),
  };
}

function validateEventOrder(events, expected) {
  if (typeof expected === 'undefined') {
    return { status: 'not_checked' };
  }
  if (!Array.isArray(expected)) {
    throw new Error('expectations.event_order must be an array');
  }
  if (events.length !== expected.length) {
    return {
      status: 'fail',
      message: `event count mismatch: expected ${expected.length}, got ${events.length}`,
    };
  }

  for (let index = 0; index < expected.length; index++) {
    const actual = events[index];
    const wanted = expected[index] || {};
    const fields = Object.keys(wanted);
    for (const field of fields) {
      const actualValue = actual[field];
      const wantedValue = wanted[field];
      if (JSON.stringify(actualValue) !== JSON.stringify(wantedValue)) {
        return {
          status: 'fail',
          message: `event ${index} field ${field} mismatch`,
          expected: wanted,
          actual,
        };
      }
    }
  }

  return { status: 'pass' };
}

function runLifecycle(context, scenario) {
  context.__scenario = scenario;

  return vm.runInContext(`
(function (scenario) {
  var events = [];
  var order = 0;
  var currentSpawnAvatars = null;
  var currentSpawnIndex = 0;
  var currentPositionAxis = null;
  var currentDirectionAttempt = null;

  function round(value) {
    return Math.round(value * 1000000) / 1000000;
  }

  function numberOr(value, fallback) {
    return typeof value === 'number' && isFinite(value) ? value : fallback;
  }

  function eventAtMs() {
    return round(__nowMs());
  }

  function recordEvent(event, data) {
    events.push({
      order: order++,
      atMs: eventAtMs(),
      event: event,
      data: data || {}
    });
  }

  function avatarId(avatar) {
    return avatar ? avatar.id : null;
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
      case 'player:leave':
        return { player: data.player.id };
      case 'score':
      case 'score:round':
        return { avatar: data.id, score: data.score, roundScore: data.roundScore };
      case 'property':
        return { avatar: data.avatar.id, property: data.property, value: data.value };
      case 'round:end':
        return { winner: avatarId(data.winner) };
      default:
        return {};
    }
  }

  function shouldRecord(game, name, data) {
    if (name === 'position' || name === 'angle') {
      return !game.world.active;
    }
    if (name === 'point') {
      return data && data.important ? true : false;
    }
    if (name === 'property') {
      return data && data.property === 'printing';
    }
    return true;
  }

  function recordSourceEvent(game, emitter, name) {
    emitter.on(name, function (data) {
      if (shouldRecord(game, name, data || {})) {
        recordEvent(name, eventData(name, data || {}));
      }
    });
  }

  function installRandomInstrumentation() {
    var originalOnRoundNew = Game.prototype.onRoundNew;
    var originalGetRandomPosition = World.prototype.getRandomPosition;
    var originalGetRandomPoint = World.prototype.getRandomPoint;
    var originalGetRandomDirection = World.prototype.getRandomDirection;
    var originalGetRandomAngle = World.prototype.getRandomAngle;
    var originalPrintManagerStart = PrintManager.prototype.start;
    var originalPrintManagerStop = PrintManager.prototype.stop;

    Game.prototype.onRoundNew = function () {
      currentSpawnAvatars = [];
      currentSpawnIndex = 0;
      for (var i = this.avatars.items.length - 1; i >= 0; i--) {
        if (this.avatars.items[i].present) {
          currentSpawnAvatars.push(this.avatars.items[i].id);
        }
      }
      try {
        return originalOnRoundNew.apply(this, arguments);
      } finally {
        currentSpawnAvatars = null;
        currentSpawnIndex = 0;
      }
    };

    World.prototype.getRandomPosition = function (radius, border) {
      var previousAxis = currentPositionAxis;
      currentPositionAxis = 0;
      try {
        return originalGetRandomPosition.call(this, radius, border);
      } finally {
        currentPositionAxis = previousAxis;
      }
    };

    World.prototype.getRandomPoint = function (margin) {
      var axis = currentPositionAxis === 0 ? 'x' : (currentPositionAxis === 1 ? 'y' : 'retry');
      __pushRandomLabel({
        site: 'spawn.position_' + axis,
        avatar: currentSpawnAvatars ? currentSpawnAvatars[currentSpawnIndex] : null
      });
      try {
        return originalGetRandomPoint.call(this, margin);
      } finally {
        __popRandomLabel();
        if (currentPositionAxis !== null) {
          currentPositionAxis += 1;
        }
      }
    };

    World.prototype.getRandomDirection = function (x, y, tolerance) {
      var previousAttempt = currentDirectionAttempt;
      currentDirectionAttempt = 0;
      try {
        return originalGetRandomDirection.call(this, x, y, tolerance);
      } finally {
        currentDirectionAttempt = previousAttempt;
        if (currentSpawnAvatars) {
          currentSpawnIndex += 1;
        }
      }
    };

    World.prototype.getRandomAngle = function () {
      var attempt = currentDirectionAttempt === null ? null : currentDirectionAttempt;
      __pushRandomLabel({
        site: attempt === null ? 'world.random_angle' : 'spawn.angle_attempt_' + attempt,
        avatar: currentSpawnAvatars ? currentSpawnAvatars[currentSpawnIndex] : null
      });
      try {
        return originalGetRandomAngle.call(this);
      } finally {
        __popRandomLabel();
        if (currentDirectionAttempt !== null) {
          currentDirectionAttempt += 1;
        }
      }
    };

    PrintManager.prototype.start = function () {
      recordEvent('print_manager:start', { avatar: this.avatar.id });
      __pushRandomLabel({
        site: 'print_manager.start_distance',
        avatar: this.avatar.id
      });
      try {
        return originalPrintManagerStart.apply(this, arguments);
      } finally {
        __popRandomLabel();
      }
    };

    PrintManager.prototype.stop = function () {
      __pushRandomLabel({
        site: 'print_manager.stop_distance',
        avatar: this.avatar.id
      });
      try {
        return originalPrintManagerStop.apply(this, arguments);
      } finally {
        __popRandomLabel();
      }
    };
  }

  function makeRoom(playerSpecs) {
    var roomConfig = scenario.source_setup && scenario.source_setup.room
      ? scenario.source_setup.room
      : {};
    return {
      name: roomConfig.name || scenario.scenario_id || scenario.id || 'lifecycle-oracle',
      players: new Collection([], 'id', true),
      config: {
        getMaxScore: function () { return numberOr(roomConfig.max_score, numberOr(roomConfig.maxScore, 10)); },
        getBonuses: function () { return []; },
        getVariable: function (name) { return name === 'bonusRate' ? 0 : undefined; }
      },
      controller: { clients: new Collection() }
    };
  }

  function addPlayers(room, playerSpecs) {
    for (var i = 0; i < playerSpecs.length; i++) {
      var spec = playerSpecs[i] || {};
      var name = spec.name || spec.id || 'p' + i;
      var player = new Player({ id: (spec.client_id || name + '-client'), active: true }, name, spec.color || '#ffffff');
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

  function firstSpecValue(spec, keys) {
    for (var i = 0; i < keys.length; i++) {
      if (typeof spec[keys[i]] !== 'undefined') {
        return spec[keys[i]];
      }
    }
    return undefined;
  }

  function applyPlayerFixtureState(avatars, playerSpecs) {
    for (var i = 0; i < playerSpecs.length && i < avatars.length; i++) {
      var spec = playerSpecs[i] || {};
      var present = firstSpecValue(spec, ['present', 'avatar_present', 'avatarPresent']);
      if (typeof present !== 'undefined') {
        if (present) {
          avatars[i].present = true;
        } else {
          avatars[i].destroy();
        }
      }
    }
  }

  function includeDeathsSnapshot() {
    return scenario.include_deaths_snapshot === true ||
      (
        scenario.lifecycle &&
        scenario.lifecycle.include_deaths_snapshot === true
      );
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
      trailPointCount: avatar.trail.points.length,
      printManager: {
        active: avatar.printManager.active,
        distance: round(avatar.printManager.distance),
        lastX: round(avatar.printManager.lastX),
        lastY: round(avatar.printManager.lastY)
      }
    };
  }

  function snapshotGame(game) {
    var hasWorld = game.world && typeof game.world === 'object';
    var snapshot = {
      size: game.size,
      started: game.started,
      inRound: game.inRound,
      worldActive: hasWorld ? game.world.active : null,
      worldBodyCount: hasWorld ? game.world.bodyCount : null,
      frameScheduled: game.frame ? true : false,
      rendered: game.rendered
    };
    if (includeDeathsSnapshot()) {
      if (game.deaths && game.deaths.items) {
        snapshot.deathCount = game.deaths.count();
        snapshot.deaths = game.deaths.items.map(function (avatar) { return avatar.id; });
      } else {
        snapshot.deathCount = null;
        snapshot.deaths = null;
      }
    }
    return snapshot;
  }

  function requiredFinite(value, field) {
    if (typeof value !== 'number' || !isFinite(value)) {
      throw new Error(field + ' must be a finite number');
    }
    return value;
  }

  function actionType(action) {
    return action && action.type ? action.type : null;
  }

  function actionAvatarId(action, actionName) {
    if (typeof action.avatar !== 'undefined') { return action.avatar; }
    if (typeof action.avatar_id !== 'undefined') { return action.avatar_id; }
    if (typeof action.avatarId !== 'undefined') { return action.avatarId; }
    throw new Error(actionName + ' action requires avatar');
  }

  function findAvatar(id) {
    for (var i = 0; i < avatars.length; i++) {
      if (String(avatars[i].id) === String(id)) {
        return avatars[i];
      }
    }
    throw new Error('avatar not found: ' + id);
  }

  function setAvatarState(action, index) {
    var avatar = findAvatar(actionAvatarId(action, 'set_avatar_state'));
    var hasX = typeof action.x !== 'undefined';
    var hasY = typeof action.y !== 'undefined';
    if (hasX || hasY) {
      avatar.setPosition(
        hasX ? requiredFinite(action.x, 'actions[' + index + '].x') : avatar.x,
        hasY ? requiredFinite(action.y, 'actions[' + index + '].y') : avatar.y
      );
    }
    if (typeof action.angle_rad !== 'undefined') {
      avatar.setAngle(requiredFinite(action.angle_rad, 'actions[' + index + '].angle_rad'));
    } else if (typeof action.angle !== 'undefined') {
      avatar.setAngle(requiredFinite(action.angle, 'actions[' + index + '].angle'));
    }
    if (typeof action.velocity !== 'undefined') {
      avatar.setVelocity(requiredFinite(action.velocity, 'actions[' + index + '].velocity'));
    }
    if (typeof action.angular_velocity !== 'undefined') {
      avatar.setAngularVelocity(requiredFinite(
        action.angular_velocity,
        'actions[' + index + '].angular_velocity'
      ));
    }
  }

  function removeAvatar(action) {
    game.removeAvatar(findAvatar(actionAvatarId(action, 'remove_avatar')));
  }

  function runAction(game, action, index) {
    switch (actionType(action)) {
      case 'advance_timers':
      case 'advance':
        __advanceTimers(requiredFinite(action.ms, 'actions[' + index + '].ms'));
        break;
      case 'set_avatar_state':
        setAvatarState(action, index);
        break;
      case 'remove_avatar':
      case 'removeAvatar':
        removeAvatar(action);
        break;
      case 'update':
        game.update(requiredFinite(action.step_ms, 'actions[' + index + '].step_ms'));
        break;
      default:
        throw new Error('unsupported lifecycle action type: ' + actionType(action));
    }
  }

  installRandomInstrumentation();
  __setRandomObserver(function (call) {
    recordEvent('random', {
      index: call.index,
      value: call.value,
      site: call.label.site,
      avatar: typeof call.label.avatar === 'undefined' ? null : call.label.avatar
    });
  });

  var playerSpecs = scenario.players || [];
  var room = makeRoom(playerSpecs);
  addPlayers(room, playerSpecs);

  var game = new Game(room);
  var avatars = game.avatars.items;
  applyPlayerFixtureState(avatars, playerSpecs);
  var warmupMs = numberOr(
    scenario.lifecycle && scenario.lifecycle.new_round_time_ms,
    numberOr(scenario.new_round_time_ms, 0)
  );
  var advances = scenario.timer_advances_ms || (
    scenario.lifecycle && scenario.lifecycle.timer_advances_ms
  ) || [0, 3000];
  var snapshots = [];

  [
    'round:new',
    'round:end',
    'clear',
    'borderless',
    'game:start',
    'game:stop',
    'player:leave',
    'end'
  ].forEach(function (name) { recordSourceEvent(game, game, name); });

  avatars.forEach(function (avatar) {
    [
      'position',
      'angle',
      'point',
      'die',
      'score',
      'score:round',
      'property'
    ].forEach(function (name) { recordSourceEvent(game, avatar, name); });
  });

  game.started = false;
  game.inRound = false;

  game.newRound(warmupMs);
  snapshots.push({
    label: 'after_new_round_call',
    atMs: eventAtMs(),
    game: snapshotGame(game),
    avatars: avatars.map(snapshotAvatar)
  });

  var actions = (scenario.lifecycle && scenario.lifecycle.actions) || scenario.actions || null;

  if (actions) {
    if (!Array.isArray(actions)) {
      throw new Error('lifecycle.actions must be an array');
    }
    for (var actionIndex = 0; actionIndex < actions.length; actionIndex++) {
      runAction(game, actions[actionIndex], actionIndex);
      snapshots.push({
        label: 'after_action_' + actionIndex + '_' + actionType(actions[actionIndex]),
        action: actions[actionIndex],
        atMs: eventAtMs(),
        game: snapshotGame(game),
        avatars: avatars.map(snapshotAvatar)
      });
    }
  } else {
    for (var i = 0; i < advances.length; i++) {
      __advanceTimers(advances[i]);
      snapshots.push({
        label: 'after_advance_' + i,
        advanceMs: advances[i],
        atMs: eventAtMs(),
        game: snapshotGame(game),
        avatars: avatars.map(snapshotAvatar)
      });
    }
  }

  return {
    scenario: scenario.scenario_id || scenario.id || 'unnamed',
    lifecycleMode: true,
    playerCount: avatars.length,
    newRoundTimeMs: warmupMs,
    timerAdvancesMs: advances,
    lifecycleActions: actions || undefined,
    events: events,
    snapshots: snapshots,
    randomCalls: __randomCalls.slice()
  };
}(__scenario))
`, context, { filename: 'lifecycle_oracle.vm.js' });
}

function main() {
  const { source, scenario } = readScenario();
  const context = makeContext(scenario);

  loadOriginalSources(context);

  const result = runLifecycle(context, scenario);
  const expectations = scenario.expectations || {};
  const validation = validateEventOrder(result.events, expectations.event_order);

  result.source = source;
  result.loadedSources = sourceFiles;
  result.expectations = { eventOrder: validation };

  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (validation.status === 'fail') {
    process.exitCode = 1;
  }
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
