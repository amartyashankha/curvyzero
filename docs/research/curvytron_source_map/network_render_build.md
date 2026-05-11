# CurvyTron Networking, Rendering, And Build Source Map

Status: source-mined on 2026-05-08.

Scope: `third_party/curvytron-reference`. This page is intentionally shallower
than the environment mechanics notes. It records the network, client rendering,
and build shape so source-fidelity work does not get surprised later.

## Short Version

Server-side game state is still the source of truth. The browser mostly sends
input changes and mirrors server events into local models and canvases.

For environment fidelity, copy these now:

- Input meaning: left is `-1`, right is `1`, no turn is `0`.
- Server tick/event order, especially `position`, `angle`, `point`, `die`,
  `property`, `score`, `score:round`, `round:new`, and `round:end`.
- Compression if comparing wire events: floats are rounded to integer
  hundredths by `Compressor.compress`.

Defer these until browser demo or pixel review:

- Exact Angular controller behavior, room UI, chat, metrics, sounds, and
  waiting overlays.
- Canvas layering, bonus sprite animation, explosion animation, and resize UI.
- Full raw-app build unless running the browser game is the task.

## Socket Protocol

The socket protocol is a JSON batch of array events. `BaseSocketClient.addEvent`
creates `[name, data, callbackId]` when a callback is present, or `[name, data]`
without one. `BaseSocketClient.sendEvents` sends `JSON.stringify(events)`.
Incoming messages are parsed in `BaseSocketClient.onMessage`; string event names
emit events, and numeric names resolve indexed callbacks.

Key refs:

- `src/shared/core/BaseSocketClient.js:105`
  `BaseSocketClient.prototype.addEvent`
- `src/shared/core/BaseSocketClient.js:131`
  `BaseSocketClient.prototype.addEvents`
- `src/shared/core/BaseSocketClient.js:187`
  `BaseSocketClient.prototype.sendEvents`
- `src/shared/core/BaseSocketClient.js:208`
  `BaseSocketClient.prototype.onMessage`
- `src/shared/core/BaseSocketClient.js:251`
  `BaseSocketClient.prototype.createCallback`

The browser opens a websocket to the current page path, using `ws://` or
`wss://`, and asks `whoami` after open. The server accepts websocket upgrades
with the `websocket` subprotocol and wraps each socket in a server-side
`SocketClient`.

Key refs:

- `src/client/core/SocketClient.js:4` `SocketClient`
- `src/client/core/SocketClient.js:35` `SocketClient.prototype.onOpen`
- `src/client/core/SocketClient.js:46` `SocketClient.prototype.onConnection`
- `src/server/core/Server.js:4` `Server`
- `src/server/core/Server.js:40` `Server.prototype.authorizationHandler`
- `src/server/core/Server.js:58` `Server.prototype.onSocketConnection`
- `src/server/core/SocketClient.js:91` `SocketClient.prototype.sendEvents`

Float-like position and angle fields are compressed for the wire:

- `src/shared/service/Compressor.js:20` `Compressor.prototype.compress`
  returns `(0.5 + value * 100) | 0`.
- `src/shared/service/Compressor.js:32` `Compressor.prototype.decompress`
  returns `value / 100`.

This only matters for exact network-message tests. State traces can keep full
source precision and avoid wire rounding unless the test is explicitly about
the socket protocol.

## Server Controllers

There are three controller levels.

`RoomsController` handles the lobby list:

- Listeners are attached in `src/server/controller/RoomsController.js:64`
  `RoomsController.prototype.attachEvents`.
- Client requests: `room:fetch`, `room:create`, `room:join`.
- Broadcasts: `room:open`, `room:close`, `room:players`, `room:game`,
  `room:config:open`.
- Main handlers: `emitAllRooms` at line 90, `onCreateRoom` at line 110,
  `onJoinRoom` at line 129.

`RoomController` handles a single room:

- Join payload is assembled in `src/server/controller/RoomController.js:100`
  `RoomController.prototype.attach`.
- Client listeners are in `RoomController.prototype.attachEvents` at line 147.
- Client requests include `room:leave`, `room:talk`, `player:add`,
  `player:remove`, `player:kick`, `room:ready`, `room:color`, `room:name`,
  `room:config:*`, and `room:launch`.
- Room broadcasts include `client:add`, `client:remove`, `room:master`,
  `room:join`, `room:leave`, `player:ready`, `player:color`, `player:name`,
  `room:game:start`, `room:kick`, `vote:new`, and `vote:close`.
- Important handlers: `onPlayerAdd` at line 395, `onReady` at line 527,
  config handlers at lines 580, 605, 623, and 644, `onLaunch` at line 663,
  `onGame` at line 703.

`GameController` is the gameplay wire surface:

- It listens for `ready` and `player:move` in
  `src/server/controller/GameController.js:142`
  `GameController.prototype.attachEvents`.
- `onMove` at line 314 takes `{avatar, move}` and calls
  `avatar.updateAngularVelocity(data.move)`.
- Server-to-client game events are emitted from:
  `onPosition` line 340, `onAngle` line 354, `onDie` line 367,
  `onBonusPop` line 381, `onBonusClear` line 396, `onScore` line 406,
  `onRoundScore` line 416, `onProperty` line 426, `onBonusStack` line 440,
  `onGameStart` line 458, `onGameStop` line 468, `onRoundNew` line 478,
  `onRoundEnd` line 488, `onClear` line 498, `onBorderless` line 508,
  and `onEnd` line 518.
- Spectator catch-up state is built in `attachSpectator` at line 199.

For source-fidelity traces, the important part is not the room UI. It is the
gameplay event shape and the fact that move changes are events into the server
avatar model.

## Client Repositories

Client repositories mirror server events into client-side models.

`RoomsRepository`:

- Attaches lobby event listeners in
  `src/client/repository/RoomsRepository.js:27`
  `RoomsRepository.prototype.attachEvents`.
- Sends `room:create` in `create` at line 74.
- Sends `room:fetch` in `start` at line 183.

`RoomRepository`:

- Attaches room event listeners in `src/client/repository/RoomRepository.js:42`
  `RoomRepository.prototype.attachEvents`.
- Sends `room:join` and builds the full client room in `join` at line 98.
- Rebuilds room config and players in `createRoom` at line 159.
- Sends player and config requests from `addPlayer` line 254,
  `removePlayer` line 268, `setReady` line 347, `setConfigOpen` line 358,
  `setConfigMaxScore` line 369, `setConfigVariable` line 380,
  `setConfigBonus` line 391, and `launch` line 399.
- Starts the local `Room` game object when `room:game:start` arrives in
  `onGameStart` at line 604.

`GameRepository`:

- Attaches gameplay events in `src/client/repository/GameRepository.js:69`
  `GameRepository.prototype.attachEvents`.
- Starts/stops the client render loop in `onGameStart` line 150 and
  `onGameStop` line 162.
- Mirrors server updates in `onProperty` line 174, `onPosition` line 189,
  `onPoint` line 206, `onAngle` line 220, `onDie` line 234,
  `onBonusPop` line 249, `onBonusClear` line 267, `onBonusStack` line 282,
  `onRoundNew` line 296, `onRoundEnd` line 307, `onClear` line 319,
  `onBorderless` line 329, `onEnd` line 340, `onLeave` line 352,
  and `onSpectate` line 364.
- `draw` at line 138 repaints only when the game has no animation frame, so
  idle state can still update visually.

This confirms the client is a display mirror for server truth. The environment
should not treat client prediction as authoritative.

## Input Path

Local player input becomes `player:move`.

- Local avatars attach `PlayerInput` in `src/client/model/Avatar.js:6`
  `Avatar`.
- Keyboard, touch, and gamepad listeners are attached in
  `src/client/model/PlayerInput.js:37`
  `PlayerInput.prototype.attachEvents`.
- `PlayerInput.prototype.resolve` at line 228 maps one active side to `-1` or
  `1`; both pressed or neither pressed becomes `false`.
- `PlayerInput.prototype.setMove` at line 258 emits a local `move` event.
- `src/client/controller/GameController.js:154`
  `GameController.prototype.onMove` sends `{avatar: id, move: value || 0}`.
- `src/server/controller/GameController.js:314`
  `GameController.prototype.onMove` applies the move to the server avatar.

So environment actions should use the resolved values `-1`, `0`, and `1`.
Browser key bindings are a UI detail.

## Rendering Path

The visual game uses four stacked canvases in
`src/client/views/game/play.html`: `background`, `bonus`, `game`, and `effect`.
CSS layers them in `src/sass/pages/_game.scss` under `.game-render`; the
`borderless` class switches the border to dashed.

Core render refs:

- `src/client/model/Game.js:43` `Game.prototype.loadDOM` creates the canvas
  wrappers.
- `src/client/model/Game.js:58` `Game.prototype.newFrame` uses
  `window.requestAnimationFrame`.
- `src/client/model/Game.js:168` `Game.prototype.draw` updates local visual
  positions, draws trails, avatars, bonus stacks, and map bonuses.
- `src/client/model/Game.js:211` `Game.prototype.drawTail` draws the latest
  trail segment onto the background canvas.
- `src/client/model/Game.js:225` `Game.prototype.drawAvatar` draws the avatar
  head canvas onto the front canvas.
- `src/client/model/Game.js:311` `Game.prototype.onResize` chooses a square
  board size from the viewport and sidebar width, then sets canvas scale.
- `src/client/core/Canvas.js:67` `Canvas.prototype.setDimension`.
- `src/client/core/Canvas.js:304` `Canvas.prototype.drawLineScaled`.
- `src/client/core/Canvas.js:340` `Canvas.prototype.round`.

Avatar rendering and interpolation:

- `src/client/model/Avatar.js:56` `Avatar.prototype.update` locally advances
  visual angle/position between server updates when unchanged.
- `src/client/model/Avatar.js:74` `Avatar.prototype.setPositionFromServer`
  sets authoritative position, marks the avatar changed, and appends a trail
  point if printing.
- `src/client/model/Avatar.js:149` `Avatar.prototype.drawHead` draws a colored
  circle.
- `src/client/model/Avatar.js:163` `Avatar.prototype.drawArrow` draws the
  local pre-round direction arrow.

Trail rendering:

- `src/client/model/Trail.js:28` `Trail.prototype.getLastSegment` returns only
  the current drawable segment, then trims old points.
- `src/client/model/Trail.js:58` `Trail.prototype.addPoint` clears the visual
  segment if the jump is larger than tolerance `1`.

Bonus rendering:

- `src/client/manager/BonusManager.js:59` `BonusManager.prototype.onLoad`
  maps `web/images/bonus.png` sprite cells to bonus class names.
- `src/client/manager/BonusManager.js:96` `BonusManager.prototype.draw`
  animates and draws active map bonuses.
- `src/client/model/bonus/MapBonus.js:39` `MapBonus.prototype.update` computes
  animated draw size.
- `src/client/model/bonus/StackedBonus.js:53`
  `StackedBonus.prototype.setEndingTimeout` starts UI warning behavior near
  expiration.

For later browser demo parity, expect visual prediction and server correction.
For training, state truth should come from the server model or Python simulator,
not from browser pixels.

## Server Mechanics Touch Points

Only a thin slice is needed here to avoid networking/rendering mistakes:

- Server game update is in `src/server/model/Game.js:37`
  `Game.prototype.update`.
- Server avatar update is in `src/server/model/Avatar.js:23`
  `Avatar.prototype.update`.
- Shared elapsed-ms movement is in `src/shared/model/BaseAvatar.js:169`
  `BaseAvatar.prototype.updateAngle` and line 186
  `BaseAvatar.prototype.updatePosition`.
- `src/server/model/Avatar.js:55` `Avatar.prototype.setPosition` emits
  `position`; line 84 `setAngle` emits `angle`; line 158 `addPoint` emits
  `point`; line 180 `die` emits `die`.
- `src/server/model/Game.js:257` `Game.prototype.onStart` emits
  `game:start`, schedules trail printing after 3000 ms, activates the world,
  then starts the base loop.
- `src/server/model/Game.js:221` `Game.prototype.onRoundEnd` resolves scores
  and emits `round:end`.

These refs explain why the client sees positions and angles as events rather
than recomputing the whole authoritative world.

## Trackers And Inspector

The `src/server/trackers/*.js` files are not gameplay protocol. They feed the
optional Inspector/Influx path.

- `src/server/core/Inspector.js:36` attaches to server client, room, and game
  events.
- `src/server/trackers/TrackerGame.js:23` listens to `round:new`,
  `game:start`, `game:stop`, and `end`.
- `src/server/trackers/TrackerClient.js:27` emits latency tracker data.

Do not pull tracker behavior into the environment. It is observability around
the app, not game semantics.

## Package, Build, And Run Shape

The reference project is an old Node web app:

- `package.json:30` has Gulp 3-era dev dependencies.
- `package.json:45` runtime dependencies are `express`, `faye-websocket`, and
  `influx`.
- `package.json:55` install script runs `bower install`.
- `bower.json:27` client dependencies include Angular 1.4.3, SoundJS, and
  `tom32i-*` browser packages.

Gulp creates generated files that are missing in this checkout:

- `recipes/server.json:3` outputs to `bin/`; `recipes/server.json:4` lists
  `src/server/dependencies.js`, shared files, server files, and launcher.
- `recipes/client.json:3` outputs to `web/js/`; `recipes/client.json:4` lists
  shared files, client files, and excludes `stressTest.js`.
- `gulpfile.js:64` `front-expose` builds `web/js/dependencies.js`.
- `gulpfile.js:75` `front-full` builds unminified `web/js/curvytron.js`.
- `gulpfile.js:83` `front-min` builds minified `web/js/curvytron.js`.
- `gulpfile.js:92` `ga` builds `web/index.html`.
- `gulpfile.js:107` `views` builds Angular views under `web/js/views`.
- `gulpfile.js:113` `server` builds `bin/curvytron.js`.
- `gulpfile.js:119` and line 127 build CSS from Sass.
- `gulpfile.js:151` `default` runs the production-style build.
- `gulpfile.js:152` `dev` runs the dev build and copies `stressTest.js`.

Runtime:

- `src/server/launcher.js:7` defaults to port `8080` and inspector disabled
  when `config.json` is missing.
- `src/server/launcher.js:13` creates `new Server({ port: config.port })`.
- `src/server/core/Server.js:21` serves static files from `web`.
- `src/server/core/Server.js:25` listens on the configured port.
- Documented run path is `npm install`, `bower install`, `gulp`,
  then `node bin/curvytron.js`.

Current known blocker, from the raw run probe: `bin/curvytron.js`,
`web/index.html`, `web/js/curvytron.js`, `web/js/dependencies.js`, and
`web/css/style.css` are not present. The safest run experiment is still a
disposable copy with an old Node runtime, not writing dependencies or build
artifacts into the reference clone.

## Fidelity Guidance

Use now for environment/source checks:

- Server model and controller event order.
- Input values after browser resolution: `-1`, `0`, `1`.
- Wire compression only when comparing socket messages.
- Client repositories as a checklist for event names and payload shapes.

Skip for first environment parity:

- Angular routing, templates, chat, sound, notifications, profile controls,
  room-list presentation, and Inspector/Influx.
- Canvas animation timing and pixel output.
- Raw browser build, unless the task is specifically browser hosting or visual
  demo validation.

Likely later browser-demo surprises:

- The client predicts visual position between server events, then corrects on
  `position`.
- Trail gaps have both server semantics and client-only visual clearing.
- Board scale is viewport-driven, so screenshots need fixed viewport sizes.
- The raw build needs old tooling: Gulp 3, Bower, old Sass tooling, and a
  generated `bin/curvytron.js`.
