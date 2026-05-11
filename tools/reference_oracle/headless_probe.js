#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { EventEmitter } = require('events');

const repoRoot = path.resolve(__dirname, '..', '..');
const referenceRoot = path.join(repoRoot, 'third_party', 'curvytron-reference');

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

function checkRawDependencyRequire() {
  const target = path.join(referenceRoot, 'src/server/dependencies.js');

  try {
    require(target);
    return {
      ok: true,
      target: relative(target),
    };
  } catch (error) {
    return {
      ok: false,
      target: relative(target),
      name: error.name,
      code: error.code || null,
      message: error.message,
      requireStack: Array.isArray(error.requireStack)
        ? error.requireStack.map(relative)
        : [],
    };
  }
}

function makeDeterministicMath() {
  const math = Object.create(Math);
  math.random = function random() {
    return 0.5;
  };
  return math;
}

function makeContext() {
  return vm.createContext({
    console,
    EventEmitter,
    Math: makeDeterministicMath(),
    Date,
    JSON,
    setTimeout(callback, delay) {
      return {
        type: 'timeout',
        delay,
        callback: callback && callback.name ? callback.name : 'anonymous',
      };
    },
    clearTimeout() {
      return null;
    },
    setInterval(callback, delay) {
      return {
        type: 'interval',
        delay,
        callback: callback && callback.name ? callback.name : 'anonymous',
      };
    },
    clearInterval() {
      return null;
    },
  });
}

function loadOriginalSources(context) {
  for (const file of sourceFiles) {
    const absolute = path.join(referenceRoot, file);
    const code = fs.readFileSync(absolute, 'utf8');
    vm.runInContext(code, context, { filename: file });
  }
}

function runForcedStep(context) {
  const probeCode = `
(function () {
  var events = [];
  var stepMs = 1000 / 60;

  function round(value) {
    return Math.round(value * 1000000) / 1000000;
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
      case 'score':
      case 'score:round':
        return { avatar: data.id, score: data.score, roundScore: data.roundScore };
      case 'bonus:pop':
        return { bonus: data.id, type: data.constructor.name, x: round(data.x), y: round(data.y) };
      case 'bonus:clear':
        return { bonus: data.id };
      case 'round:end':
        return { winner: avatarId(data.winner) };
      default:
        return {};
    }
  }

  function record(emitter, name) {
    emitter.on(name, function (data) {
      events.push({ event: name, data: eventData(name, data || {}) });
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
      printManager: {
        active: avatar.printManager.active,
        distance: round(avatar.printManager.distance),
        lastX: round(avatar.printManager.lastX),
        lastY: round(avatar.printManager.lastY)
      }
    };
  }

  var room = {
    name: 'oracle-probe',
    players: new Collection([], 'id', true),
    config: {
      getMaxScore: function () { return 10; },
      getBonuses: function () { return []; },
      getVariable: function (name) { return name === 'bonusRate' ? 0 : undefined; }
    },
    controller: { clients: new Collection() }
  };

  function addPlayer(name, color) {
    var player = new Player({ id: name + '-client', active: true }, name, color);
    room.players.add(player);
    return player;
  }

  addPlayer('p0', '#ff0000');
  addPlayer('p1', '#00ff00');

  var game = new Game(room);
  var avatars = game.avatars.items;
  var a0 = avatars[0];
  var a1 = avatars[1];

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
  });

  game.started = true;
  game.inRound = true;
  game.world.activate();

  a0.setPosition(20, 40);
  a0.setAngle(0);
  a0.updateVelocities();
  a0.printManager.start();

  a1.setPosition(60, 40);
  a1.setAngle(Math.PI);
  a1.updateVelocities();
  a1.printManager.start();

  events.length = 0;

  a0.updateAngularVelocity(-1);
  a1.updateAngularVelocity(1);
  game.update(stepMs);

  return {
    scenario: 'forced_two_player_turn_step',
    feasible: true,
    stepMs: round(stepMs),
    game: {
      size: game.size,
      started: game.started,
      inRound: game.inRound,
      borderless: game.borderless,
      deathCount: game.deaths.count(),
      deaths: game.deaths.items.map(function (avatar) { return avatar.id; }),
      roundWinner: avatarId(game.roundWinner),
      gameWinner: avatarId(game.gameWinner),
      worldBodyCount: game.world.bodyCount
    },
    avatars: [
      snapshotAvatar(a0, -1),
      snapshotAvatar(a1, 1)
    ],
    events: events
  };
}())
`;

  return vm.runInContext(probeCode, context, { filename: 'headless_probe.vm.js' });
}

function main() {
  const output = {
    rawDependencyRequire: checkRawDependencyRequire(),
    vmProbe: null,
    loadedSources: sourceFiles,
  };

  const context = makeContext();
  loadOriginalSources(context);
  output.vmProbe = runForcedStep(context);

  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
}

main();
