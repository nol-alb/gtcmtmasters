// tone-basic-envelope

let osc;
let env;
let isOn = false;
let hasStarted = false;

function setup() {
  createCanvas(windowWidth, windowHeight);
  background(0);

  env = new Tone.AmplitudeEnvelope({
    attack: 1.5,
    decay: 0.21,
    sustain: 0.5,
    release: 1.2
  }).toDestination();

  // Connect our oscillator with envelope
  osc = new Tone.Oscillator(440, "sine").connect(env);
}

function draw() {
  background(0);

  fill(255);
  textSize(130);
  textAlign(CENTER);

  if (isOn && hasStarted) {
    text('ON', width/2, height/2);
  } else if (!isOn && hasStarted) {
    text('OFF', width/2, height/2)
  } else {
    text('Click to start', width/2, height/2)
  }
}

function mousePressed() {
  hasStarted = true;
  osc.start();
}

function keyPressed() {
  if (keyCode === UP_ARROW && hasStarted) {
    env.triggerAttack();
    isOn = !isOn;
  }

  if (keyCode === DOWN_ARROW && hasStarted) {
    env.triggerRelease();
    isOn = !isOn;
  }
}