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

function runProbe(context) {
  return vm.runInContext(`
(function () {
  var stepMs = 100;
  var startGap = 0.05;

  function round(value) {
    return Math.round(value * 1000000) / 1000000;
  }

  function avatarId(avatar) {
    return avatar ? avatar.id : null;
  }

  function makeRoom() {
    return {
      name: 'border-probe',
      players: new Collection([], 'id', true),
      config: {
        getMaxScore: function () { return 10; },
        getBonuses: function () { return []; },
        getVariable: function (name) { return name === 'bonusRate' ? 0 : undefined; }
      },
      controller: { clients: new Collection() }
    };
  }

  function addPlayer(room, name, color) {
    var player = new Player({ id: name + '-client', active: true }, name, color);
    room.players.add(player);
    return player;
  }

  function snapshotAvatar(avatar) {
    return {
      id: avatar.id,
      name: avatar.name,
      x: round(avatar.x),
      y: round(avatar.y),
      angle: round(avatar.angle),
      velocityX: round(avatar.velocityX),
      velocityY: round(avatar.velocityY),
      radius: round(avatar.radius),
      alive: avatar.alive,
      present: avatar.present,
      printing: avatar.printing,
      trailPointCount: avatar.trail.points.length,
      bodyNum: avatar.body.num,
      bodyCount: avatar.bodyCount
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
      worldBodyCount: game.world.bodyCount
    };
  }

  function eventData(name, data) {
    switch (name) {
      case 'position':
        return { avatar: data.id, x: round(data.x), y: round(data.y) };
      case 'die':
        return {
          avatar: data.avatar.id,
          killer: data.killer ? data.killer.id : null,
          old: typeof data.old === 'undefined' ? null : data.old
        };
      case 'round:end':
        return { winner: avatarId(data.winner) };
      case 'borderless':
        return { value: data ? true : false };
      default:
        return {};
    }
  }

  function record(events, emitter, name) {
    emitter.on(name, function (data) {
      events.push({ event: name, data: eventData(name, data || {}) });
    });
  }

  function forceState(game) {
    var target = game.avatars.items[0];
    var control = game.avatars.items[1];
    var y = game.size / 2;
    var startX = game.size - target.radius - startGap;

    game.started = true;
    game.inRound = true;
    game.world.activate();

    target.setPosition(startX, y);
    target.setAngle(0);
    target.updateVelocities();

    control.setPosition(game.size / 2, y);
    control.setAngle(Math.PI);
    control.updateVelocities();

    return target;
  }

  function classify(game, before, after, events) {
    var died = events.some(function (event) { return event.event === 'die'; });
    var positions = events.filter(function (event) {
      return event.event === 'position' && event.data.avatar === before.id;
    });
    var crossedRight = positions.some(function (event) { return event.data.x > game.size; });

    if (died || !after.alive) {
      return 'dies';
    }

    if (crossedRight && after.alive && after.x === 0) {
      return 'wraps';
    }

    if (after.alive) {
      return 'continues';
    }

    return 'unknown';
  }

  function runCase(mode, borderless) {
    var events = [];
    var room = makeRoom();
    var game, target, before, after;

    addPlayer(room, 'p0', '#ff0000');
    addPlayer(room, 'p1', '#00ff00');

    game = new Game(room);

    record(events, game, 'borderless');
    record(events, game, 'round:end');
    target = forceState(game);
    record(events, target, 'position');
    record(events, target, 'die');

    events.length = 0;

    if (borderless) {
      game.setBorderless(true);
    }

    before = snapshotAvatar(target);

    game.avatars.items.forEach(function (avatar) {
      avatar.updateAngularVelocity(0);
    });
    game.update(stepMs);

    after = snapshotAvatar(target);

    return {
      mode: mode,
      result: classify(game, before, after, events),
      stepMs: stepMs,
      startGap: startGap,
      game: snapshotGame(game),
      targetBefore: before,
      targetAfter: after,
      events: events
    };
  }

  return {
    probe: 'border_behavior',
    target: 'avatar 1 crosses the right boundary',
    cases: [
      runCase('normal', false),
      runCase('borderless', true)
    ],
    loadedSourceCount: ${sourceFiles.length}
  };
}())
`, context, { filename: 'border_probe.vm.js' });
}

function main() {
  const context = makeContext();
  loadOriginalSources(context);

  process.stdout.write(`${JSON.stringify(runProbe(context), null, 2)}\n`);
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
