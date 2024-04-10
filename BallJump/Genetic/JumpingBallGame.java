
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

public class JumpingBallGame extends JPanel implements KeyListener {
    private final int WIDTH = 600;
    private final int HEIGHT = 400;
    private final Color WHITE = Color.WHITE;
    private final Color BLACK = Color.BLACK;
    private final Color RED = Color.RED;
    private final int BALL_RADIUS = 10;
    private final int PLAYER_RADIUS = 20;
    private final int PLAYER_JUMP = 12;
    private final double PLAYER_GRAVITY = 0.6;

    private Player player;
    private List<Ball> balls;

    private Timer timer;

    public JumpingBallGame() {
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        setBackground(WHITE);
        setFocusable(true);
        addKeyListener(this);

        player = new Player(WIDTH / 2, HEIGHT - PLAYER_RADIUS * 2, PLAYER_RADIUS);
        balls = new ArrayList<>();

        timer = new Timer();
        timer.scheduleAtFixedRate(new GameLoop(), 0, 1000 / 30);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;

        player.draw(g2d);
        for (Ball ball : balls) {
            ball.draw(g2d);
        }
    }

    @Override
    public void keyPressed(KeyEvent e) {
        if (e.getKeyCode() == KeyEvent.VK_SPACE) {
            player.jump();
        }
    }

    @Override
    public void keyTyped(KeyEvent e) {}

    @Override
    public void keyReleased(KeyEvent e) {}

    private class Player {
        private int x, y, radius;
        private double velY;

        public Player(int x, int y, int radius) {
            this.x = x;
            this.y = y;
            this.radius = radius;
            this.velY = 0;
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
            g2d.setColor(RED);
            g2d.fillOval(x - radius, y - radius, radius * 2, radius * 2);
        }
    }

    private class Ball {
        private int x, y, radius, speed;

        public Ball(int x, int y, int radius, int speed) {
            this.x = x;
            this.y = y;
            this.radius = radius;
            this.speed = speed;
        }

        public void update() {
            x -= speed;
        }

        public void draw(Graphics2D g2d) {
            g2d.setColor(BLACK);
            g2d.fillOval(x - radius, y - radius, radius * 2, radius * 2);
        }
    }

    private class GameLoop extends TimerTask {
        @Override
        public void run() {
            player.update();

            for (int i = 0; i < balls.size(); i++) {
                Ball ball = balls.get(i);
                ball.update();
                if (ball.x < -BALL_RADIUS) {
                    balls.remove(i);
                    i--;
                }
            }

            if (new Random().nextInt(150) == 0) {
                addBall();
            }

            checkCollision();

            repaint();
        }
    }

    private void addBall() {
        int ballSize = 15;
        int ballSpeed = new Random().nextInt(6) + 3;
        balls.add(new Ball(WIDTH, HEIGHT - ballSize * 2, ballSize, ballSpeed));
    }

    private void checkCollision() {
        for (Ball ball : balls) {
            Rectangle ballRect = new Rectangle(ball.x - ball.radius, ball.y - ball.radius, ball.radius * 2, ball.radius * 2);
            Rectangle playerRect = new Rectangle(player.x - player.radius, player.y - player.radius, player.radius * 2, player.radius * 2);
            if (ballRect.intersects(playerRect)) {
                System.exit(0);
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Jumping Ball Game");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);
            frame.getContentPane().add(new JumpingBallGame());
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
