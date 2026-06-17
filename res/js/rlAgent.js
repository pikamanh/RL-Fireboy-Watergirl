import { currentLevel } from "./helpers.js";

const ACTION_ENDPOINT = "/rl/action";
const params = new URLSearchParams(window.location.search);
const enabled = params.get("rl") === "1";

let latestActions = [0, 0];
let pending = false;

function applyAction(player, action) {
    player.keys.pressed.left = action === 1 || action === 4;
    player.keys.pressed.right = action === 2 || action === 5;

    if ((action === 3 || action === 4 || action === 5) && player.isOnBlock && !player.rampBlocked) {
        player.velocity.y = -4.35;
        player.keys.pressed.up = true;
    } else {
        player.keys.pressed.up = false;
    }
}

function playerState(player) {
    return {
        x: player.position.x,
        y: player.position.y,
        vx: player.velocity.x,
        vy: player.velocity.y,
        onGround: player.isOnBlock,
    };
}

export function isRlEnabled() {
    return enabled;
}

export function resetRlAgent() {
    latestActions = [0, 0];
    pending = false;
}

export function applyRlActions(players) {
    if (!enabled) return;

    players.forEach((player, index) => {
        applyAction(player, latestActions[index] ?? 0);
    });
}

export function requestRlAction(players) {
    if (!enabled || pending || players.length < 2) return;

    pending = true;
    fetch(ACTION_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            level: Number(currentLevel),
            players: players.map(playerState),
        }),
    })
        .then((response) => response.json())
        .then((data) => {
            if (Array.isArray(data.actions)) {
                latestActions = data.actions;
            }
        })
        .catch((error) => {
            console.warn("RL action request failed", error);
        })
        .finally(() => {
            pending = false;
        });
}
