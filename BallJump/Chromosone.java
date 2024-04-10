
public class Chromosone {
    // t = d/s
    
    double MUTATIONRATE = 0.05;
    double MUTATIONRANGE = 5;
    public double time;
    
    public Chromosone(){
        time = Math.random() * 20;
    }
    public Chromosone(Chromosone A, Chromosone B){
        time = (A.time + B.time)/2;
    }

    public void mutate(){
        if (Math.random() < MUTATIONRATE){
            double randomTime = ((2 * Math.random() * MUTATIONRANGE) - MUTATIONRANGE)/2;
            time += randomTime;
        }
    }    
    public String toString(){
        return "" + time;
    }
}
