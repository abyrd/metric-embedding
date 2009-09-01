import processing.opengl.*;
import unlekker.util.*;
import unlekker.geom.*;
import unlekker.data.*;
import processing.video.*;
import hypermedia.net.*;   // UDP
import java.nio.ByteBuffer;

static final int MODE_SURFACE = 0, MODE_POINTS = 1, MODE_LINES = 2;
int displayMode = MODE_POINTS;

MovieMaker mm;
boolean recording = false;

UDP udp;
String mesg;
float  mesgVal;


// 2-dimensional array of 3D vector objects to contain our mesh
Vec3 mesh[][];

// 2d mesh of 3D vectors for tweening
Vec3 targetMesh[][];

// 2d array of SPT reults (travel times)
int  ttimes[][];

PImage  texImage;       // holds the texture map image
boolean texMap = false; // whether or not texture mapping is turned on
boolean drawPaths = false;

// num will decide the mesh dimensions
int num;  

// boolean flags indicating whether or not we should
// output STL or PDF
boolean doSTL=false,doPDF=false;

// use the ALT key for interface
boolean modDown=false;

// translation the Z direction
float xUnit, yUnit;
float xTrans, yTrans, zTrans;
float rotX, rotY, amplitude;
float velX, velY;
float autoRot = 0;

void setup() {
  // changing window size has little effect on FPS
  //size(800, 600, OPENGL);
  size(1280, 1024, OPENGL);
  frameRate(30);
  texImage = loadImage("PDX2rot.jpg");
  // texmap = loadImage("apple.jpg");
  // allocate space for the mesh and fill it with empty vectors  
  num=100;
  mesh=new Vec3[num][num];
  targetMesh=new Vec3[num][num];
  ttimes = new int[num][num];
  meshPlane(mesh);
  meshPlane(targetMesh);
  
  xUnit=width/(float)num;
  yUnit=height/(float)num;

  rotX=30;
  rotY=45;
  amplitude=1;
  xTrans =  width  / 2;
  yTrans =  height / 2;
  zTrans = -width  / 2;

  // add a MouseWheelListener so we can use the mouse wheel
  // to zoom with
  frame.addMouseWheelListener(new MouseWheelInput());
  udp = new UDP( this, 6000 );
  udp.listen(true);
  // textFont(createFont("ProggySmall", 22));
  textFont(loadFont("pixel.vlw"));
}

void receive( byte[] data, String ip, int port ) throws IOException {
  // be careful of byte order
  // Java always reads in network, python will send in machine unless you specify in struct.pack
  // ByteBuffers and DataInputStreams are good for unpacking data
  int row, col, t;
  float x, y, z;
  byte b;
  ByteBuffer bb = ByteBuffer.wrap(data);
  // InputStream in = new ByteArrayInputStream(data);
  // DataInputStream din = new DataInputStream(in);
  b = bb.get();
  // Receive travel times at gridpoints
  if (b == (byte)'T') {
    while (bb.hasRemaining()) {
      row = bb.get() & 0xFF;
      col = bb.get() & 0xFF;
      t = (int) (bb.get() & 0xFF);
      if (row < 100 && col < 100) ttimes[row][col] += t;
      // java supports only signed chars      
      // ttimes[row][col] = 255;
      // targetMesh[row][col].y = (int)(data[i+2] & 0xFF) / 255.0;
    }
  // Receive new coordinates for gridpoints
  } else if (b == (byte)'P') {
    while (bb.hasRemaining()) {
      row = bb.get() & 0xFF;
      col = bb.get() & 0xFF;
      x = bb.getFloat();
      y = bb.getFloat();
      z = bb.getFloat();
      if (row < 100 && col < 100) {
        targetMesh[row][col].x = x;
        targetMesh[row][col].z = y;
        targetMesh[row][col].y = z;
      }
      // targetMesh[row][col].x = (data[i+2] & 0xFF) - 127;
      // targetMesh[row][col].z = (data[i+3] & 0xFF) - 127;
      // targetMesh[row][col].y = (data[i+4] & 0xFF) - 127;
    }
  // Receive 3d scatterplot irrespective of grid
  } else if (b == (byte)'S') {
    while (bb.hasRemaining()) {
      t = bb.getInt(); // point number
      x = bb.getFloat();
      y = bb.getFloat();
      z = bb.getFloat();
      row = t / num;  // assign up to n^2 points to independent gridpoints
      col = t % num;  // 
      targetMesh[row][col].x = x;
      targetMesh[row][col].z = y;
      targetMesh[row][col].y = z;
    }
  }
}

void draw() {
  background(0);
  //background(32);
  
  if(doSTL) beginRaw("unlekker.data.STL","UnlekkerMesh"+frameCount+".stl");

  // perspective();
  // ortho();
  // if you do lights before translate, they will move with the point of view
  // better illuminates paths
  // lights();

  // translate to center of screen and use mouse movement
  // to rotate the scene
  translate(xTrans, yTrans, zTrans);
  rotateX(rotX);
  rotateY(rotY);
  rotX += velX;
  rotY += velY + autoRot;
  // veolcities decay arithmetically with time
  velX *= 0.7;
  velY *= 0.7;
  
  // if you do lights after translations, they are fixed rel to the objects
  // looks pretty in contre-jour.
  lights();
  
  // fade out travel time information
  for (int i=0; i<num; i++) {
    for (int j=0; j<num; j++) ttimes[i][j] *= 0.96;
  }  
  // tween shape of object
  Vec3 path;
  for (int i=0; i<num; i++) {
    for (int j=0; j<num; j++) {
      // println(mash, targetMesh);
      path = new Vec3(targetMesh[i][j]);
      path.sub(mesh[i][j]);
      path.mult(0.05);
      mesh[i][j].add(path);
    }
  }  
  if (displayMode == MODE_SURFACE) drawMesh();
  else {
    for (int i=0; i<num-1; i++) {
      for (int j=0; j<num; j++) {
        int t = ttimes[i][j];
        stroke(100+t, 155+t, 100+t);
        // if (displayMode == MODE_POINTS) point(mesh[i][j].x, mesh[i][j].y * amplitude, mesh[i][j].z);
        if (displayMode == MODE_POINTS && mesh[i][j].y != 0) point(mesh[i][j].x, mesh[i][j].y * amplitude, mesh[i][j].z);
        // else line(mesh[i][j].x, mesh[i][j].y * amplitude, mesh[i][j].z, mesh[i+1][j].x, mesh[i+1][j].y * amplitude, mesh[i+1][j].z);
        else if (mesh[i][j].y != 0) line(mesh[i][j].x, mesh[i][j].y * amplitude, mesh[i][j].z, 0, 0, 0);
      }
    }  
  }
  
  if (recording) mm.addFrame();  
  
  if(doSTL || doPDF) {
    endRaw();
    doSTL=false;
    doPDF=false;
  }
}

// different meshes can be triggered by the keyboard.
// STL and PDF output is also available

void keyPressed() {
  if      (key=='1') meshPlane(targetMesh);
  else if (key=='2') meshNoise(targetMesh);
  else if (key=='3') meshSineWave(targetMesh);
  else if (key=='S') doSTL=true;
  // toggle texture mapping
  else if (key=='m') texMap = ! texMap;
  // select surface, point, or line display
  else if (key=='s') displayMode = MODE_SURFACE ;
  else if (key=='p') displayMode = MODE_POINTS ;
  else if (key=='l') displayMode = MODE_LINES ;
  // toggle recording
  else if(key=='r') {
    if (recording) {
      recording = false;
      mm.finish();
    } else {
      mm = new MovieMaker(this, width, height, "display.mov", 30, MovieMaker.MOTION_JPEG_A, MovieMaker.BEST);
      recording = true;
    }
  }
  // fake incoming SPT results to test
  else if(key=='f') fakeTravelTimes();
  // check for ALT key
  else if (key == CODED && keyCode==ALT) modDown=true;
}

void keyReleased() {
  // check for ALT key to see if it has been released
  if (key == CODED && keyCode==ALT) modDown=false;
}

void mouseDragged() {
  // if ALT key is down, translate surface object in window
  if(modDown) {
      xTrans += (mouseX-pmouseX) ;
      yTrans += (mouseY-pmouseY) ;
  } else {
    // for left clicks, rotational acceleration
    if (mouseButton == LEFT) { 
      velX -= (mouseY-pmouseY) * 0.002;
      velY += (mouseX-pmouseX) * 0.002;
      // rotX=((float)-mouseY/(float)width)*2*PI;
      // rotY=((float)-mouseX/(float)height)*2*PI;
    // for right clicks, change the amplitude
    } else if (mouseButton == RIGHT) {
      // y axis adjusts amplitude of graph 
      amplitude += (mouseY-pmouseY) * 0.005; // ((float)mouseY/(float)height)*height;
      // x axis adjusts coloration
      mesgVal = mouseX / (float)width * 255;
    }
  }
  
}

// drawMesh is a custom method that takes care of drawing the
// mesh using beginShape() / endShape()
void drawMesh() {
  float y;
  float fade;
  
  // if (frameCount % 30 == 0) fakeTravelTimes();  
 
  noStroke(); 
  // texture mapping cuts framerate by two thirds
  // use 0...1 (not pixel numbers) for intra-image coordinates
  if (texMap) textureMode(NORMALIZED);  
  for(int i=0; i<num-1; i++) {
    beginShape(QUAD_STRIP);
    // fill(255, 255); // i/float(num));
    if (texMap) texture(texImage);
    for(int j=0; j<num; j++) {
      // make the graph get darker from one side to the other
      // could look better with fog
      
      // using alpha on the object only cuts off a few fps
      // fill(100, z, mesgVal, sqrt(i/float(num)) * 255);  
      // fill(200, 255, 200);  
      y = mesh[j][i].y * amplitude;
      // fade to black in same direction as light falls for more drama
      if (texMap) {
        fade = 32 + (ttimes[i][j] * 2);
        fill(fade, fade, fade) ;  
        // extra coords specify corresponding point in the texture image
        vertex(i, y, j, i/float(num), j/float(num));
      } else {
        fade = float(i) / float(num);
        fade += (ttimes[i][j]) / 255.0 * 3;
        fill(100 * fade, y * fade, mesgVal * fade);  
        vertex(i, y, j);
      }
      
      y = mesh[j][i+1].y * amplitude;
      // fade to black in same direction as light falls for more drama
      if (texMap) {
        fade = 32 + (ttimes[i+1][j] * 2);
        fill(fade, fade, fade);  
        // extra coords specify corresponding point in the texture image
        vertex(i+1, y, j, (i+1)/float(num), j/float(num));
      } else {
        fade = float(i+1) / float(num);
        fade += (ttimes[i+1][j]) / 255.0 * 3;
        fill(100 * fade, y * fade, mesgVal * fade);  
        vertex(i+1, y, j);
      }

//      y = mesh[j][i+1].y * amplitude;
//      if (texMap) fade = 1;
//      else fade = float(i+1) / float(num);
//      fade += (ttimes[j][i+1]) / 255.0 * 3;
//      fill(100 * fade, y * fade, mesgVal * fade);  
//      // extra coords specify corresponding point in the texture image
//      if (texMap) vertex(mesh[j][i+1].x, y, mesh[j][i+1].z, j/float(num), (i+1)/float(num));
//      else        vertex(mesh[j][i+1].x, y, mesh[j][i+1].z);
  
    }
    endShape();    
  }
  if (drawPaths) {
    for (int i=0; i<num; i++) {
      for (int j=0; j<num; j++) {
        stroke(255);
        line(mesh[j][i].x, -10, mesh[j][i].z, 
             targetMesh[j][i].x, -10, targetMesh[j][i].z);
      }
    }
  }
  if (false) { // frameCount % 60 == 0) {
    textMode(SCREEN);  // put text at screen surface, not in 3D
    fill(255);         // make text white
    text(frameRate + " fps\n", 20, 30);
  }
  // calling text() on screen every frame slows down by 50%!
  // use bitmap fonts to speed up? no. 
  // textMode(SCREEN);  // put text at screen surface, not in 3D
  // fill(255);         // make text white
  // text(frameRate + " FPS", 20, 30);   // build onscreen message
}

void fakeTravelTimes() {
  int low   = int(random(num-10));
  int range = int(random(5, 10));
  int high = low + range;
  int direction = int(random(2));
  for (int i=low; i<high; i++) {
    for (int j=0; j<num; j++) {
      int val = int(sin((i - low) / float(range) * PI) * (num - j) * 3);
      if (direction == 0) ttimes[i][j] += val;
      else ttimes[j][i] += val;      
    }
  }
}

// sets the mesh to an even grid plane
// must call this to initialize a mesh, as it creates the constituent vertex vectors
void meshPlane(Vec3[][] mesh) {
  for(int i=0; i<num; i++) 
    for(int j=0; j<num; j++) 
      mesh[i][j]=new Vec3((i-num/2)*10,0,(j-num/2)*10);
}

// sets the mesh data to a Perlin noisefield
void meshNoise(Vec3[][] mesh) {
  float x,y,a,aD,b,bD,val;

  // initialize parameters for the noise field. 
  // "a" is our position in the X direction of the noise.
  // "b" is our position in the X direction of the noise.
  // "aD" and "bD" are used to traverse the noise field.
  a=random(1000);
  b=random(1000);
  aD=1.0/random(50,150);
  bD=1.0/random(50,150);

  // set amplitude and noiseDetail for noise field
  noiseDetail((int)random(4,8),random(0.4,0.9));

  for(int i=0; i<num; i++) 
    for(int j=0; j<num; j++) {

      // calculate height as a function of 2D noise
      val=(noise(a+aD*(float)i,b+bD*(float)j)-0.5)*2;

      x=((float)i-num/2)*xUnit;;
      y=((float)j-num/2)*yUnit;
      mesh[i][j].set(x,val,y);
    }
}

// sets the mesh data to a 3D plot of two sine waves
void meshSineWave(Vec3[][] mesh) {
  float x,y,a,aD,b,bD;
  float val;

  // set random starting values and wavelengths
  // for our sine wave landscape
  a=radians(random(360));
  b=radians(random(360));
  aD=radians(random(0.1,12));
  bD=radians(random(0.1,12));

  for(int i=0; i<num; i++) 
    for(int j=0; j<num; j++) {
      x=((float)i-num/2)*xUnit;;
      y=((float)j-num/2)*yUnit;
      // calculate height as a function of two sine curves
      val=sin(a+aD*(float)i) * sin(b+bD*(float)j);
      mesh[i][j].set(x,val,y);

    }
}

// convenience class to listen for MouseWheelEvents and
// use it for that classic "zoom" effect
 
class MouseWheelInput implements MouseWheelListener{

  void mouseWheelMoved(MouseWheelEvent e) {
    int step=e.getWheelRotation();
    // with modifier key, wheel adjusts automatic rotation speed
    if (modDown) autoRot -= (step) * 0.001;
    // without mod key, it zooms in and out on the scene
    else zTrans=zTrans+step*-50;
  }
 
}
