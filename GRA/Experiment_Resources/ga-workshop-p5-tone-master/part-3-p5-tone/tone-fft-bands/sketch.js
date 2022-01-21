// tone-fft-bands

// reverse-arp.wav from this pack: https://samples.landr.com/packs/emotional-sounds-vol-1
// zimbalom.wav from freesound.org
// Code heavily adapted from https://glitch.com/edit/#!/dfpi-audio-fft-2?path=sketch.js%3A99%3A3

let player;
let analyser;
let isPlaying = false;

function setup() {
  createCanvas(windowWidth, windowHeight);
  background(0);

  // setup text
  textSize(28);
  textAlign(CENTER, CENTER);

  analyser = new AudioEnergy();

  // setup our player with an audio file
  player = new Tone.Player('reverse-arp.wav');
  // player = new Tone.Player('zimbalom.wav');
  player.loop = true;
  player.autostart = false;

  player.connect(analyser);
  player.connect(Tone.Master); // renamed toDestination() in newer versions of Tone
}

function draw() {
 
  if (isPlaying) {

    // Ensure everything is loaded first
    if (!player || !analyser) return;

    // Black background
    background(0);

    // Update the analyser
    analyser.update();
    noStroke();

    // Draw fft
    if (isPlaying) {
      const diameter = width * 0.2;
      const freqRanges = [
        
        { name: 'bass', color: '#355C7D' },
        { name: 'lowMid', color: '#6C5B7B' },
        { name: 'mid', color: '#C06C84' },
        { name: 'highMid', color: '#F67280' },
        { name: 'treble', color: '#F8B195' }
      ];

      for (let i = 0; i < freqRanges.length; i++) {
        const hexColor = freqRanges[i].color;
        const name     = freqRanges[i].name;
        const energy   = analyser.getEnergy(name);
        const scale    = map(energy, -100, -30, 0, 1);

        fill(hexColor);

        const finalDiameter = diameter + scale * diameter
        ellipse(i * (width * 0.2) + diameter/2, height/2, finalDiameter, finalDiameter);

        fill(21, 21, 21);
        text(name, i * (width * 0.2) + diameter/2, height/2);
      }
    }
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