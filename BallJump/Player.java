import java.awt.*;

public class Player {
    private static final int X = JumpingBallGameGenetic.WIDTH / 2;
    public static final int PLAYER_RADIUS = 20;
    private static final int Y =  JumpingBallGameGenetic.HEIGHT - PLAYER_RADIUS * 2;
    public double x;
    public double y;
    public int radius;
    public final static int HEIGHT = 400;
    private final int PLAYER_JUMP = 10;
    private final double PLAYER_GRAVITY = 0.5;
    private double velY;
    private Chromosone dna;
    private int balldoge = 0;
    private boolean dead ;
    public Player() {
        this.x = X;
        this.y = Y;
        this.radius = PLAYER_RADIUS;
        this.velY = 0;
        this.dna = new Chromosone();
        dead = false;
    }
    public void die(){
        dead = true;
    }
    public boolean IsDead(){
        return dead;
    }
    public Player(Player a, Player b){
        this.dna = new Chromosone(a.dna,b.dna);
        this.velY = 0;
        dead = false;
        this.x = X;
        this.radius = PLAYER_RADIUS;
        this.y = Y;
    }
    public boolean willJump(double ball_x, double speed){
        double distance = ball_x - x;
        double time = distance/speed;
        System.out.println("Distance: " + distance + " Speed: " + speed + " Time: " + time );

        return Math.abs(time - dna.time) < 1;
    }
    public void mutate(){
        dna.mutate();
    }
    public int getfitness(){
        return balldoge;
    }
    public void doge(){
        balldoge++;
        if(balldoge > 50)
            System.out.println("50 JUMP WINNER: " + dna);
    }

    public void jump() {
        if (y == HEIGHT - PLAYER_RADIUS * 2) {
            velY = -PLAYER_JUMP;
        }
    }

    public void update() {
        velY += PLAYER_GRAVITY;
        y += velY;

        if (y >= HEIGHT - PLAYER_RADIUS * 2) {
            y = HEIGHT - PLAYER_RADIUS * 2;
            velY = 0;
        }
    }

    public void draw(Graphics2D g2d) {
        g2d.setColor(Color.RED);
        g2d.fillOval((int)x - PLAYER_RADIUS, (int)y - PLAYER_RADIUS, PLAYER_RADIUS * 2, PLAYER_RADIUS * 2);
    }
    public String toString(){
        return "Fit:" + balldoge + "\t TIME: " + dna;
    }
}
