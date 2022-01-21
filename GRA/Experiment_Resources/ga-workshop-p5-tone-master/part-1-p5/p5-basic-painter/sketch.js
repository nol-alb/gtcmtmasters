// p5-basic-painter

let slider;
let colorPicker;

function setup() {
  createCanvas(windowWidth, windowHeight);
  background(21);

  // setup slider
  slider = createSlider(0, 100, 20);
  slider.position(40, 40);
  slider.style('width', '80px');

  // setup colorpicker
  colorPicker = createColorPicker('#ed225d');
  colorPicker.position(200, 40);
}

function draw() {
  noStroke();
  fill(0);
  rect(0, 0, width, 80);

  fill(255);
  textSize(12);
  text('brush size (' + slider.value() + ')', 40, 30);
  text('color', 200, 30);
  text('press any key to clear', 600, 45);
}

function mouseDragged() {
  fill(colorPicker.color());
  ellipse(mouseX, mouseY, slider.value(), slider.value());
}

function keyPressed() {
  background(21);
}