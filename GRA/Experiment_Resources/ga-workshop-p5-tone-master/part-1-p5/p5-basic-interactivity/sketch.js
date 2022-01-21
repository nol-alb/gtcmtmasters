// p5-basic-interactivity
let diameter;
let r, g, b;

function setup() {
  createCanvas(windowWidth, windowHeight);
  diameter = 40;
  pickRandomColor();
}

function draw() {
  background(250, 200, 255);
  noStroke();
  fill(r, g, b);
  
  ellipse(mouseX, mouseY, diameter, diameter);
}

function mousePressed() {
  diameter = random(40, 180);
}

function keyPressed() {
  pickRandomColor();
}

function pickRandomColor() {
  r = random(255);
  g = random(255);
  b = random(255);
}