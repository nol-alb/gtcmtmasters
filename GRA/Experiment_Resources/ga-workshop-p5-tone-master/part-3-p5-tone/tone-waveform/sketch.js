// tone-waveform

let synth;
let bgCol;
let size;
let waveform;

const AMinorScale = ['A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4'];

function setup() {
  createCanvas(windowWidth, windowHeight);

  bgCol = color(207, 236, 207);
  background(bgCol);
  textSize(22);
  
  // power of 32, in the range [32, 32768]
  size = 1024;

  synth = new Tone.Synth();
  waveform = new Tone.Waveform(size).toDestination();
  synth.connect(waveform);
  
  noFill();
  strokeWeight(2);
}

function draw() {
  background(bgCol);
  text('ASDFJKL keys!', 20, 40);

  // draw the wave
  const waveArray = waveform.getValue();
  const bandSize = width / size;

  beginShape();

  for (let i = 0; i < waveArray.length; i++) {
    curveVertex(bandSize * i, map(waveArray[i], -1, 1, height, 0));
  }
  endShape();
}

function mousePressed() {
  hasStarted = true;
  Tone.start();
}

function keyPressed() {
  switch (key) {
    case ('a'): 
      synth.triggerAttackRelease(AMinorScale[0], '16n'); 
      bgCol = color(207, 236, 207);
      break;
    
    case ('s'): 
      synth.triggerAttackRelease(AMinorScale[1], '16n');
      bgCol = color(154, 206, 223); 
      break;
    
    case ('d'): 
      synth.triggerAttackRelease(AMinorScale[2], '16n'); 
      bgCol = color(221, 212, 232);
      break;
    
    case ('f'): 
      synth.triggerAttackRelease(AMinorScale[3], '16n'); 
      bgCol = color(253, 202, 162);
      break;
    
    case ('j'): 
      synth.triggerAttackRelease(AMinorScale[4], '16n');
      bgCol = color(219, 213, 185); 
      break;

    case ('k'): 
      synth.triggerAttackRelease(AMinorScale[5], '16n'); 
      bgCol = color(179, 226, 221);
      break;
    
    case ('l'): 
      synth.triggerAttackRelease(AMinorScale[6], '16n'); 
      bgCol = color(148, 168, 208);
      break;
  }
}