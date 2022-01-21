// tone-player

let player;
let isPlaying = false;

function setup() {
  createCanvas(windowWidth, windowHeight);

  // setup text
  textSize(64);
  textAlign(CENTER, CENTER);

  // setup our player with an audio file
  player = new Tone.Player('kick.wav').toDestination();
  player.loop = true;
  player.autostart = false;
}

function draw() {
  background(250, 200, 250);
  if (isPlaying) {
    fill(0);
    text('KICK', width/2, height/2);
  }
}

function keyPressed() {
  if (player && player.loaded) {
    // console.log('player loaded');

    if (isPlaying) {
      // we were playing, so we'll stop
      isPlaying = !isPlaying;
      player.stop();
    } else {
      // we weren't playing, so we'll start
      isPlaying = !isPlaying;
      player.start();
    }
  }
}