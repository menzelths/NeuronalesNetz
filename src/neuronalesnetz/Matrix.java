/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuronalesnetz;

import java.util.function.Function;

/**
 *
 * @author menze
 */
public class Matrix {

    double[] werte = null;
    int spaltenZahl = 0, zeilenZahl = 0;
    
    public static Double sigmoid(Double x){
        return 1/(1+Math.exp(-x));
    }
    

    public Matrix(double[] werte, int spalten) {
        this.werte = werte;
        this.spaltenZahl = spalten;
        this.zeilenZahl = werte.length / spalten;
    }
    
    public Matrix(int zeilen, int spalten){
        this.werte=new double[zeilen*spalten];
        this.spaltenZahl=spalten;
        this.zeilenZahl=zeilen;
    }
    
    public Matrix(int zeilen, int spalten,Function<Double,Double> funktion){
        this.werte=new double[zeilen*spalten];
        this.spaltenZahl=spalten;
        this.zeilenZahl=zeilen;
        for (int i=0;i<werte.length;i++){
            werte[i]=funktion.apply(werte[i]);
        }
    }

    public double get(int zeile, int spalte) {
        return werte[spaltenZahl * zeile + spalte];
    }

    public void set(int zeile, int spalte, double wert) {
        werte[spaltenZahl * zeile + spalte] = wert;
    }

    public int getZeilen() {
        return zeilenZahl;
    }

    public int getSpalten() {
        return spaltenZahl;
    }

    public Matrix transponiert() {
        double[] werteNeu = new double[werte.length];
        Matrix m = new Matrix(werteNeu, this.zeilenZahl);
        for (int y = 0; y < this.zeilenZahl; y++) {
            for (int x = 0; x < this.spaltenZahl; x++) {
                m.set(x, y, this.get(y, x));
            }
        }
        return m;
    }
    
    public Matrix punktWeise(Function<Double,Double> funktion){
        double[] werteNeu=new double[werte.length];
        for (int i=0;i<werte.length;i++){
            werteNeu[i]=funktion.apply(werte[i]);
        }
        return new Matrix(werteNeu,spaltenZahl);
    }

    public Matrix skalar(double d) {
        double[] werteNeu = new double[werte.length];
        for (int i = 0; i < werte.length; i++) {
            werteNeu[i] = werte[i] * d;
        }
        return new Matrix(werteNeu, this.spaltenZahl);
    }

    public Matrix mal(Matrix other) {
        int anzahlSpaltenNeu = other.spaltenZahl;
        int anzahlZeilenNeu = this.zeilenZahl;

        double[] werteNeu = new double[anzahlSpaltenNeu * anzahlZeilenNeu];

        for (int y = 0; y < this.zeilenZahl; y++) {
            for (int x = 0; x < other.spaltenZahl; x++) {
                double summe = 0;
                for (int i = 0; i < this.spaltenZahl; i++) {
                    summe += this.get(y, i) * other.get(i, x);
                }
                werteNeu[x + y * anzahlSpaltenNeu] = summe;
            }
        }
        return new Matrix(werteNeu, anzahlSpaltenNeu);
    }
    
    //public Matrix punktweises
    
    public Matrix plus(Matrix other){
        double[] werteNeu=new double[werte.length];
        for (int i=0;i<werte.length;i++){
            werteNeu[i]=werte[i]+other.werte[i];
        }
        return new Matrix(werteNeu,this.spaltenZahl);
    }
    
        public Matrix minus(Matrix other){
        double[] werteNeu=new double[werte.length];
        for (int i=0;i<werte.length;i++){
            werteNeu[i]=werte[i]-other.werte[i];
        }
        return new Matrix(werteNeu,this.spaltenZahl);
    }

    @Override
    public String toString() {
        String s = "";
        for (int y = 0; y < this.zeilenZahl; y++) {
            for (int x = 0; x < this.spaltenZahl; x++) {
                s += this.get(y, x) + "\t";
            }
            s += "\n";
        }
        return s;
    }
}
