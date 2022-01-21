// p5-text

let robotoMono;

function preload() {
  robotoMono = loadFont('RobotoMono.ttf');
}

function setup() {
  createCanvas(windowWidth, windowHeight);
  
  // setup our font so we can use it
  textFont(robotoMono);
  textSize(64);
  textAlign(CENTER, CENTER);
}

function draw() {
  background(0);
  fill(0, 255, 255);
  text(key, width/2, height/2);
}