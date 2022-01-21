// p5-simple-shapes

function setup() {
  createCanvas(windowWidth, windowHeight);
  background(0);
}

function draw() {
  stroke(255);


  // by default rect() starts drawing from the top left corner,
  // you can change this with rectMode():
  // https://p5js.org/reference/#/p5/rectMode
  // rect(posX, posY, width, height)
  rect(40, 120, 120, 40);



  // by default ellipse() starts drawing from the center
  // you can change this with ellipseMode():
  // https://p5js.org/reference/#/p5/ellipseMode
  // ellipse(centerX, centerY, width, height)
  ellipse(80, 200, 30, 30);



  // arc(xCoord, yCoord, width, height, angle to start in radians, angle to stop in radians)
  arc(120, 55, 50, 50, 0, HALF_PI);



  // line(x1, y1, x2, y2); - coords for two points
  line(160, 20, 85, 75);



  // triangle(x1, y1, x2, y2, x3, y3) - coords for all three points
  triangle(240, 200, 150, 230, 260, 160);



  // quad(x1, y1, x2, y2, x3, y3, x4, y4) - coords for all four points
  quad(38, 31, 86, 20, 69, 63, 30, 76);
  
}