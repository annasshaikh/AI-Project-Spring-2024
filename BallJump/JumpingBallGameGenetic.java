
import javax.swing.*;

import java.awt.*;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

public class JumpingBallGameGenetic extends JPanel  {
    public final static int WIDTH = 600;
    public final static int HEIGHT = 400;
    private final Color WHITE = Color.WHITE;
    private final int POPULATIONSIZE = 2;

    private Population players;
    private Ball ball;

    private Timer timer;

    public JumpingBallGameGenetic() {
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        setBackground(WHITE);
        setFocusable(true);
        //addKeyListener(this);
        players = new Population(POPULATIONSIZE,true);
        respawn();

        timer = new Timer();
        timer.scheduleAtFixedRate(new GameLoop(), 0, 1000 / 30);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        ball.draw(g2d);
        players.draw(g2d);
    }
    /*
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

    */
    private class GameLoop extends TimerTask {
        @Override
        public void run() {

            ball.update();
            if (ball.x < -ball.radius) 
                respawn();
            players.update(ball);
            checkCollision();
            if (players.allDead() && ball.x < 0)
                players = players.bread();
            repaint();
        }
    }


    private void respawn() {
        int ballSize = 15;
        double ballSpeed = (Math.random()* 5) + 3;
        ball = new Ball(WIDTH, HEIGHT - ballSize * 2, ballSpeed);
        players.ballPass();
        System.out.println("Ball Speed: " + ballSpeed + " Players Left: " + players.left());
    }

    private void checkCollision() {
        players.checkCollision(ball);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Jumping Ball Game");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);
            frame.getContentPane().add(new JumpingBallGameGenetic());
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
