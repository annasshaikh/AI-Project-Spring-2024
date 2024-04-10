
import java.awt.*;
import java.util.ArrayList;

public class Population {
    private ArrayList<Player> players;
    private int size;
    private int Numdead;
    public Population(int size, boolean initilizePopulation){
        players = new ArrayList<Player>();
        this.size = size;
        this.Numdead = 0;
        if (initilizePopulation)
            initilizePopulation();
    }
    public int left(){
        return size - Numdead;
    }
    private void initilizePopulation(){
        for (int i = 0; i < size; i++)
            players.add(new Player());
    }
    public void addPlayer(Player p){
        players.add(p);
    }
    public void draw(Graphics2D g2d ){
        for (Player p : players)
            if (!p.IsDead())
                p.draw(g2d);
    }
    public boolean allDead(){
        for (Player p : players)
            if (!p.IsDead())
                return false;
        return true;
    }
    public void checkCollision(Ball ball) {
        for (Player player : players) {
            if (player.IsDead())
                continue;
            Rectangle ballRect = new Rectangle((int)ball.x - ball.radius, (int)ball.y - ball.radius, ball.radius * 2, ball.radius * 2);
            Rectangle playerRect = new Rectangle((int)player.x - player.radius, (int)player.y - player.radius, player.radius * 2, player.radius * 2);
            if (ballRect.intersects(playerRect)) {
                player.die();
                Numdead++;
            }
        }
    }
    public void ballPass(){
        for (Player p : players)
            if (!p.IsDead())
                p.doge();
    }
    public void update(Ball b){
        for (Player p : players)
            if (!p.IsDead()){
                p.update();
                if (p.willJump(b.x, b.speed))
                    p.jump();
            }

    }
    public Population bread(){
        if (!allDead())
            return null;
        summary();
        Population childrens = new Population(size,false);
        ArrayList<Player> breadingPool = new ArrayList<Player>();
        for (Player p : players)
            for (int i = 0; i < Math.pow(p.getfitness(), (int)Math.cbrt(size)); i++)
                breadingPool.add(p);
        int breadingPoolSize = breadingPool.size();
        for (int i = 0; i < size; i++){
            Player PA = breadingPool.get((int)(Math.random() * breadingPoolSize));
            Player PB = breadingPool.get((int)(Math.random() * breadingPoolSize));
            Player c = new Player(PA, PB);
            c.mutate();
            childrens.addPlayer(c);
        }
        return childrens;
    }
    private void summary(){
        
        System.out.println("\n\nSUMMARY");
        for (Player p : players){
            System.out.println(p);
        }
    }
}
