import lejos.nxt.Button;
import lejos.nxt.ButtonListener;
import lejos.nxt.Motor;
import lejos.robotics.navigation.DifferentialPilot;
import lejos.util.Delay;
import java.lang.Math;


public class DiffDrv {
  DifferentialPilot pilot;
  
  double distance = 90.0;
  double degree = 250;
  double radians = degree * Math.PI / 180;
  double radius = distance *  radians;
  
  public void ahead() {
    Delay.msDelay(500);
    pilot.travel(distance);
    pilot.stop();
  }

  public void leftArc() {
    Delay.msDelay(500);
    pilot.travelArc(radius, distance);
    pilot.stop();
  }

  public void rightArc() {
    Delay.msDelay(500);
    pilot.travelArc(-radius, distance);
    pilot.stop();
  }
  
  
  public static void main(String[] args) {
    DiffDrv traveler = new DiffDrv();
    
    double wheelDiameter = 5.6f;
    double trackWidth = 17.6f;
    traveler.pilot = new DifferentialPilot(wheelDiameter , trackWidth, Motor.A, Motor.B, false);
    
    while (true) {
      if (Button.ENTER.isDown())traveler.ahead();
      if (Button.LEFT.isDown()) traveler.leftArc();
      if (Button.RIGHT.isDown()) traveler.rightArc();
      if (Button.ESCAPE.isDown()) break;
    }
  }
}
