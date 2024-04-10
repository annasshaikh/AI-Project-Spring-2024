import java.awt.*;
public class Ball {
    public double x, y, speed;
    public int radius;
    
    public final int BALL_RADIUS = 20;
    public Ball(double x, double y, double speed) {
        this.x = x;
        this.y = y;
        this.radius = BALL_RADIUS;
        this.speed = speed;
    }
    public void update() {
        x -= speed;
    }
    public double getDistance(int x){
        return this.x - x;
    }
    public double getSpeed(){
        return speed;
    }
    public void draw(Graphics2D g2d) {
        g2d.setColor(Color.BLACK);
        g2d.fillOval((int)x - radius, (int)y - radius, radius * 2, radius * 2);
    }
}