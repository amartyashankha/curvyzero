from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_node_json_probe(source: str) -> dict[str, object]:
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    result = subprocess.run(
        ["node", "-e", source],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_js_player_input_reduction_and_release():
    # This loads the original PlayerInput and client GameController.onMove.
    # The DOM/EventEmitter are minimal shims, so this proves source control
    # reduction and client wire-value conversion, not real browser dispatch.
    payload = _run_node_json_probe(
        r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const repoRoot = process.cwd();
const referenceRoot = path.join(repoRoot, 'third_party', 'curvytron-reference');
const windowListeners = {};

function BrowserEventEmitter() {
  this.__events = {};
}

BrowserEventEmitter.prototype.on = function on(name, callback) {
  if (!this.__events[name]) {
    this.__events[name] = [];
  }
  this.__events[name].push(callback);
  return this;
};

BrowserEventEmitter.prototype.off = function off(name, callback) {
  const listeners = this.__events[name] || [];
  this.__events[name] = listeners.filter(function keep(listener) {
    return listener !== callback;
  });
  return this;
};

BrowserEventEmitter.prototype.removeListener = BrowserEventEmitter.prototype.off;

BrowserEventEmitter.prototype.emit = function emit(name, detail) {
  const listeners = (this.__events[name] || []).slice();
  const event = { type: name, detail: detail };
  listeners.forEach(function call(listener) {
    listener.call(this, event);
  }, this);
  return this;
};

const context = vm.createContext({
  JSON,
  EventEmitter: BrowserEventEmitter,
  __windowListeners: windowListeners,
  window: {
    addEventListener(name, callback) {
      if (!windowListeners[name]) {
        windowListeners[name] = [];
      }
      windowListeners[name].push(callback);
    },
    removeEventListener(name, callback) {
      const listeners = windowListeners[name] || [];
      windowListeners[name] = listeners.filter(function keep(listener) {
        return listener !== callback;
      });
    }
  },
  gamepadListener: {
    on() {},
    off() {},
    start() {},
    stop() {}
  }
});

vm.runInContext(
  'function AbstractController() {} AbstractController.prototype = {};',
  context,
  { filename: 'client_controller_test_shim.vm.js' }
);

[
  'src/client/model/PlayerInput.js',
  'src/client/controller/GameController.js'
].forEach(function load(file) {
  vm.runInContext(
    fs.readFileSync(path.join(referenceRoot, file), 'utf8'),
    context,
    { filename: file }
  );
});

const result = vm.runInContext(`
(function () {
  var avatar = { id: 7 };
  var input = new PlayerInput(avatar, [37, 39]);
  var inputMoves = [];
  var wireMoves = [];
  var noDuplicateEmits = [];
  var controller = {
    client: {
      addEvent: function (name, data) {
        wireMoves.push({ name: name, avatar: data.avatar, move: data.move });
      }
    }
  };

  input.on('move', function (event) {
    inputMoves.push(event.detail.move === false ? 'false' : event.detail.move);
    GameController.prototype.onMove.call(controller, event);
  });

  function dispatch(type, keyCode) {
    (__windowListeners[type] || []).forEach(function call(callback) {
      callback({ keyCode: keyCode });
    });
  }

  function expectNoEmit(label, callback) {
    var before = inputMoves.length;
    callback();
    noDuplicateEmits.push({
      label: label,
      before: before,
      after: inputMoves.length
    });
  }

  dispatch('keydown', 37);
  expectNoEmit('repeat left keydown while left-only stays left', function () {
    dispatch('keydown', 37);
  });
  dispatch('keyup', 37);
  expectNoEmit('repeat left keyup while neutral stays neutral', function () {
    dispatch('keyup', 37);
  });

  dispatch('keydown', 39);
  expectNoEmit('repeat right keydown while right-only stays right', function () {
    dispatch('keydown', 39);
  });
  dispatch('keyup', 39);
  expectNoEmit('repeat right keyup while neutral stays neutral', function () {
    dispatch('keyup', 39);
  });

  dispatch('keydown', 37);
  dispatch('keydown', 39);
  expectNoEmit('repeat right keydown while both keys stay neutral', function () {
    dispatch('keydown', 39);
  });
  dispatch('keyup', 39);
  dispatch('keyup', 37);

  dispatch('keydown', 39);
  dispatch('keydown', 37);
  expectNoEmit('repeat left keydown while both keys stay neutral', function () {
    dispatch('keydown', 37);
  });
  dispatch('keyup', 37);
  dispatch('keyup', 39);

  return {
    inputMoves: inputMoves,
    wireMoves: wireMoves,
    noDuplicateEmits: noDuplicateEmits,
    keyboardListenerCounts: {
      keydown: (__windowListeners.keydown || []).length,
      keyup: (__windowListeners.keyup || []).length
    },
    finalActive: input.active.slice(),
    finalInputMove: input.move === false ? 'false' : input.move
  };
}())
`, context, { filename: 'player_input_reduction_probe.vm.js' });

process.stdout.write(JSON.stringify(result));
"""
    )

    assert payload["keyboardListenerCounts"] == {"keydown": 1, "keyup": 1}
    assert payload["inputMoves"] == [
        -1,
        "false",
        1,
        "false",
        -1,
        "false",
        -1,
        "false",
        1,
        "false",
        1,
        "false",
    ]
    assert [event["name"] for event in payload["wireMoves"]] == ["player:move"] * 12
    assert [event["avatar"] for event in payload["wireMoves"]] == [7] * 12
    assert [event["move"] for event in payload["wireMoves"]] == [
        -1,
        0,
        1,
        0,
        -1,
        0,
        -1,
        0,
        1,
        0,
        1,
        0,
    ]
    assert all(
        item["before"] == item["after"] for item in payload["noDuplicateEmits"]
    )
    assert payload["finalActive"] == [False, False]
    assert payload["finalInputMove"] == "false"


def test_js_player_move_reaches_server_update_angular_velocity():
    # This uses the original server GameController constructor, attachEvents,
    # callback wrapper, and onMove with a fake client/player/avatar. It proves
    # synchronous server-side delivery after a player:move emit, not Socket.IO.
    payload = _run_node_json_probe(
        r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { EventEmitter } = require('events');

const repoRoot = process.cwd();
const referenceRoot = path.join(repoRoot, 'third_party', 'curvytron-reference');
const timers = [];

const context = vm.createContext({
  JSON,
  EventEmitter,
  __timers: timers,
  setTimeout(callback, delay) {
    const timer = { callback: callback, delay: delay, active: true };
    timers.push(timer);
    return timer;
  },
  clearTimeout(timer) {
    if (timer) {
      timer.active = false;
    }
    return null;
  }
});

[
  'src/shared/Collection.js',
  'src/shared/service/Compressor.js',
  'src/server/core/SocketGroup.js',
  'src/server/controller/GameController.js'
].forEach(function load(file) {
  vm.runInContext(
    fs.readFileSync(path.join(referenceRoot, file), 'utf8'),
    context,
    { filename: file }
  );
});

const result = vm.runInContext(`
(function () {
  var calls = [];
  var socketEvents = [];
  var pingStarts = 0;
  var phase = 'setup';
  var game = new EventEmitter();
  var bonusManager = new EventEmitter();
  var avatar = new EventEmitter();
  var player;
  var client;
  var controller;

  avatar.id = 42;
  avatar.bonusStack = new EventEmitter();
  avatar.updateAngularVelocity = function (move) {
    calls.push({ move: move, phase: phase, index: calls.length });
  };

  player = {
    id: 42,
    avatar: avatar,
    getAvatar: function () {
      return avatar;
    }
  };

  client = new EventEmitter();
  client.id = 9;
  client.players = new Collection([player]);
  client.pingLogger = {
    start: function () { pingStarts += 1; },
    stop: function () {}
  };
  client.addEvent = function (name, data) {
    socketEvents.push({ name: name, data: data });
  };
  client.addEvents = function (events) {
    socketEvents.push({ name: 'batch', data: events });
  };
  client.isPlaying = function () {
    return true;
  };

  game.bonusManager = bonusManager;
  game.room = { controller: { clients: new Collection([client]) } };
  game.started = false;
  game.isReady = function () {
    return false;
  };
  game.getLoadingAvatars = function () {
    return new Collection();
  };

  controller = new GameController(game);

  [-1, 0, 1].forEach(function emitMove(move) {
    phase = 'during player:move ' + move;
    client.emit('player:move', { avatar: 42, move: move });
    phase = 'after player:move ' + move;
  });

  return {
    calls: calls,
    pingStarts: pingStarts,
    waitingDelay: __timers[0] ? __timers[0].delay : null,
    socketEvents: socketEvents,
    attachedClientCount: controller.clients.count()
  };
}())
`, context, { filename: 'server_player_move_probe.vm.js' });

process.stdout.write(JSON.stringify(result));
"""
    )

    assert payload["attachedClientCount"] == 1
    assert payload["pingStarts"] == 1
    assert payload["waitingDelay"] == 30000
    assert payload["calls"] == [
        {"move": -1, "phase": "during player:move -1", "index": 0},
        {"move": 0, "phase": "during player:move 0", "index": 1},
        {"move": 1, "phase": "during player:move 1", "index": 2},
    ]
    assert payload["socketEvents"] == [{"name": "game:spectators", "data": 0}]
