// tone-basic-synth

let synth;

function setup() {
  createCanvas(windowWidth, windowHeight);
  background(0);
  noStroke();

  synth = new Tone.Synth().toDestination();
}

function draw() {
  background(0);
  if ( mouseIsPressed ) {
    fill(255, 200, 255);
    ellipse(mouseX, mouseY, 100, 100);
  }
}

function mousePressed() {
  synth.triggerAttackRelease("C4", "8n");
}