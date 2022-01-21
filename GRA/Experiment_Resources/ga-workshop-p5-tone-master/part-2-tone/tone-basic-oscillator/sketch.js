// tone-basic-oscillator

let osc;
let isOn = false;

function setup() {
  createCanvas(windowWidth, windowHeight);
  background(0);

  // Four basic types of waves: 
  // sine, square, sawtooth, triangle
  // Sine wave at 440hz = A4
  osc = new Tone.Oscillator(440, "sine").toDestination();
}

function draw() {
  background(0);

  fill(255);
  textSize(130);
  textAlign(CENTER);

  if (isOn) {
    text('ON', width/2, height/2);
  } else {
    text('OFF', width/2, height/2)
  }
}

function mousePressed() {
  if (isOn) {
    osc.stop();
  } else {
    osc.start();
  }

  isOn = !isOn;
}