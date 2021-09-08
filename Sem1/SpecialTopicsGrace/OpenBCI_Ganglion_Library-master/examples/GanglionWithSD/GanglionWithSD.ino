

/*
  Uesd to test the basic functionality of the Ganglion Board
  Targets a Simblee. LIS2DH, MCP3912, AD5621 on board

  Made by Joel Murphy, Leif Percifield, and AJ Keller for OpenBCI, Inc. 2016


        MUST CHANGE THE SPI PINS IN THE variants.h FILE Lines 88-92
        #define SPI_INTERFACE        NRF_SPI0
        #define PIN_SPI_SS           (26u)  //(6u)
        #define PIN_SPI_MOSI         (18u)  //(5u)
        #define PIN_SPI_MISO         (15u)  //(3u)
        #define PIN_SPI_SCK          (16u)  //(4u)
 */

#include <OpenBCI_Ganglion_Library.h>
#include <OBCI_Ganglion_SD.h>


boolean SDfileOpen = false; // Set true by SD_Card_Stuff.ino on successful file open

unsigned long thisTime;
unsigned long thatTime;

void setup() {

  ganglion.initialize();
  attachPinInterrupt(MCP_DRDY, MCP_ISR, LOW);

}


void loop() {

    if(ganglion.MCP_dataReady){

      ganglion.processData();


      if(SDfileOpen) {  // Verify the SD file is open
        // Write to the SD card
        writeDataToSDcard(ganglion.sampleCounter);
      }
    }

    ganglion.blinkLED();

    if(ganglion.eventSerial()){
      sdProcessChar(ganglion.inChar); // check for an SD related command
    }

    if(ganglion.testingImpedance){
      ganglion.testImpedance();
    }



} // end of loop


int MCP_ISR(uint32_t dummyPin) { // gotta have a dummyPin...

      ganglion.MCP_dataReady = true;
      ganglion.sampleCounter++;

      return 0; // gotta return nothing, somehow...
    }
