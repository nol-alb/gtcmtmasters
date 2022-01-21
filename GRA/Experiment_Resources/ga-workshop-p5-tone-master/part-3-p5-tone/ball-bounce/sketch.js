// p5-ball-bounce// tone-synth-keys

let synth;
let bgR, bgG, bgB;
let randomNote;
let xPos;
let yPos;
let xSpeed = 2;
let ySpeed = 3;
let radius = 20;
let hasStarted = false;

const AMinorScale = ['A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4'];

function setup() {
  createCanvas(windowWidth, windowHeight);
  noStroke();

  // set up our synth
  synth = new Tone.Synth().toDestination();

  // set up our background RGB
  bgR = random(255);
  bgG = random(255);
  bgB = random(255);

  xPos = random(windowWidth);
  yPos = random(windowHeight);
}

function draw() {
  background(bgR, bgG, bgB);
  fill(255, 255, 255);

  ellipse(xPos, yPos, radius * 2, radius * 2);

  // update our ball location
  if (hasStarted) {
    xPos += xSpeed;
    yPos += ySpeed;
  }

  if (xPos > width - radius || xPos < radius) {
    handleCollision();
    xSpeed = -xSpeed;
  }
  if (yPos > height - radius || yPos < radius) {
    handleCollision();
    ySpeed = -ySpeed;
  }
}

function handleCollision() {
  // choose random colors for bg
  bgR = random(255);
  bgG = random(255);
  bgB = random(255);

  // choose a random note from the scale
  randomNote = int(random(0, AMinorScale.length));
  // console.log(randomNote);

  // play the random note
  synth.triggerAttackRelease(AMinorScale[randomNote], '16n'); 
}

function mousePressed() {
  hasStarted = true;
  Tone.start();
}