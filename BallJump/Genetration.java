public class Genetration {
    Population CurrentPopulation;
    Population PrePopulation;
    public Genetration(Population prePopulation, int size){
        this.size = size;
        this.PrePopulation = prePopulation;
    }
    public Genetration(int size){
        this.PrePopulation = null;
        this.CurrentPopulation = new Population(size);
    }

    private bread(){
        PrePopulation = CurrentPopulation;
        CurrentPopulation = CurrentPopulation.bread();
    }
}
