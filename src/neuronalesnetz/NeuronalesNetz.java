/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuronalesnetz;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;

/**
 *
 * @author menze
 */

// Nach dem Buch von Tariq Rashid - Neuronale Netze selbst programmieren
public class NeuronalesNetz {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Random zufall = new Random();
        // TODO code application logic here

        final double lernrate = 0.3;
        int[] knotenProSchicht = {784, 100, 10}; // Definiert die Anzahl der Knoten pro Schicht, erst die Eingabeschicht, dann alle anderen 
        List<Matrix> netz = new ArrayList<>();
       // for (int epoche = 0; epoche < 1; epoche++) {
            
            // legt zufällige Übergangsmatrizen an
            for (int i = 1; i < knotenProSchicht.length; i++) {
                double standardabweichung = Math.pow(knotenProSchicht[i], -.5);
                netz.add(new Matrix(knotenProSchicht[i], knotenProSchicht[i - 1], x -> (Double) (zufall.nextGaussian() * standardabweichung)));
                //System.out.println(netz.get(netz.size()-1));
            }

            // erstellt einen zufälligen Eingabevektor
            Matrix schichtAusgabe = new Matrix(knotenProSchicht[0], 1, x -> zufall.nextGaussian());

            //System.out.println("Eingang:\n"+schichtAusgabe);
            // Führt die Berechnungen vom Eingang zum Ausgang durch
            for (Matrix gewichtung : netz) {
                schichtAusgabe = (gewichtung.mal(schichtAusgabe)).punktWeise(Matrix::sigmoid);
            }

            // Training
            // zufälligen Eingabevektor erstellen, dieser ist später ein Trainingsdatum
            schichtAusgabe = new Matrix(knotenProSchicht[0], 1, x -> zufall.nextGaussian());

            // zufälligen Vergleichsvektor erstellen, später aus Trainingsdaten
            Matrix sollWert = new Matrix(knotenProSchicht[knotenProSchicht.length - 1], 1, x -> zufall.nextGaussian());
            
            int datensatzZähler=0;
            try {
                BufferedReader br = new BufferedReader(new FileReader("C:\\Users\\menze\\Documents\\MNIST-Daten\\mnist_train.csv"));
                
                
                String zeile = "";
                while ((zeile = br.readLine()) != null) {
                    datensatzZähler++;
                    String[] datenwerte = zeile.split(",");
                    int zahl = Integer.parseInt(datenwerte[0]);
                    double[] werte = new double[10];
                    for (int i = 0; i < 10; i++) {
                        werte[i] = 0.01;
                    }
                    werte[zahl] = 0.99;
                    // Ausgabevektor festlegen
                    sollWert = new Matrix(werte, 1);

                    double[] eingangsWerte = new double[datenwerte.length - 1];
                    for (int i = 1; i < datenwerte.length; i++) {
                        eingangsWerte[i - 1] = Double.parseDouble(datenwerte[i]) / 255 * 0.99 + 0.01;
                    }

                    schichtAusgabe = new Matrix(eingangsWerte, 1);

                    List<Matrix> zwischenergebnis = new ArrayList<>();
                    zwischenergebnis.add(schichtAusgabe); // Eingabewerte aufnehmen

                    // Berechnung mit Eingabevektor durchführen
                    for (Matrix gewichtung : netz) {
                        schichtAusgabe = ((gewichtung.mal(schichtAusgabe)).punktWeise(Matrix::sigmoid));
                        zwischenergebnis.add(schichtAusgabe);
                    }

                    Matrix fehlerVektor = sollWert.minus(schichtAusgabe);

                    // Fehler der Schichten berechnen
                    List<Matrix> fehlerZwischenschichten = new ArrayList<>();
                    fehlerZwischenschichten.add(fehlerVektor);
                    for (int i = netz.size() - 1; i > 0; i--) {
                        fehlerVektor = netz.get(i).transponiert().mal(fehlerVektor);
                        fehlerZwischenschichten.add(fehlerVektor);
                    }

                    for (int aktuelleSchicht = zwischenergebnis.size() - 1; aktuelleSchicht > 0; aktuelleSchicht--) {
                        // Fehler berechnen

                        fehlerVektor = fehlerZwischenschichten.get(fehlerZwischenschichten.size() - aktuelleSchicht);

                        Matrix fehlerVektorMitSigmoide = new Matrix(fehlerVektor.zeilenZahl, 1);
                        for (int i = 0; i < fehlerVektor.zeilenZahl; i++) {

                            fehlerVektorMitSigmoide.set(i, 0, lernrate * fehlerVektor.get(i, 0) * zwischenergebnis.get(aktuelleSchicht).get(i, 0) * (1 - zwischenergebnis.get(aktuelleSchicht).get(i, 0)));
                        }
                        Matrix differenzMatrix = fehlerVektorMitSigmoide.mal(zwischenergebnis.get(aktuelleSchicht - 1).transponiert());

                        //System.out.println("Davor:\n" + netz.get(aktuelleSchicht - 1));
                        netz.set(aktuelleSchicht - 1, netz.get(aktuelleSchicht - 1).plus(differenzMatrix));
                       
                    }
                }
            } catch (Exception e) {

            }
        //}
        System.out.println("Fertig! Trainiert mit "+datensatzZähler+" Datensätzen!");
        int offset=2;
        try {
                BufferedImage bi=new BufferedImage(100*(28+offset),100*56,BufferedImage.TYPE_INT_ARGB);
                BufferedReader br = new BufferedReader(new FileReader("C:\\Users\\menze\\Documents\\MNIST-Daten\\mnist_test.csv"));
                Graphics g=bi.getGraphics();
                g.setColor(Color.WHITE);
                g.fillRect(0,0,100*(28+offset),100*56);
                g.setFont(new Font("Arial",Font.PLAIN,20));
                
                String zeile = "";
                int zaehlerKorrekt=0;
                int zaehlerGesamt=0;
                while ((zeile = br.readLine()) != null) {
                    String[] datenwerte = zeile.split(",");
                    int zahl = Integer.parseInt(datenwerte[0]);
                    double[] werte = new double[10];
                    for (int i = 0; i < 10; i++) {
                        werte[i] = 0.01;
                    }
                    werte[zahl] = 0.99;
                    // Ausgabevektor festlegen
                    sollWert = new Matrix(werte, 1);

                    double[] eingangsWerte = new double[datenwerte.length - 1];
                    for (int i = 1; i < datenwerte.length; i++) {
                        int farbwert=255-Integer.parseInt(datenwerte[i]);
                        eingangsWerte[i - 1] = Double.parseDouble(datenwerte[i]) / 255 * 0.99 + 0.01;
                        g.setColor(new Color(farbwert,farbwert,farbwert));
                        int x=(zaehlerGesamt%100)*(28+offset)+i%28;
                        int y=(zaehlerGesamt/100)*56+i/28;
                        g.drawLine(x, y, x+1, y+1);
                    }
                    

                    schichtAusgabe = new Matrix(eingangsWerte, 1);
                    
                                for (Matrix gewichtung : netz) {
                schichtAusgabe = (gewichtung.mal(schichtAusgabe)).punktWeise(Matrix::sigmoid);
            }
                                //System.out.print(zahl);
                                
                                
                                double max=0;
                                int index=-1;
                                for (int i=0;i<schichtAusgabe.zeilenZahl;i++){
                                    double aktuelleZahl=schichtAusgabe.get(i,0);
                                    if (aktuelleZahl>max){
                                        max=aktuelleZahl;
                                        index=i;
                                    }
                                    
                                }
                                g.setColor(Color.red);
                                if (index==zahl){
                                    zaehlerKorrekt++;
                                    g.setColor(Color.green);
                                } 
                                g.drawString(""+index, (zaehlerGesamt%100)*(28+offset)+9, (zaehlerGesamt/100)*56+45);
                                zaehlerGesamt++;
                }
                ImageIO.write(bi, "png", new File("D:\\test.png"));
                
                
                System.out.println("Trefferquote: "+((double)zaehlerKorrekt/zaehlerGesamt)+" bei "+zaehlerGesamt+" Datensätzen");
        } catch(Exception e){
            e.printStackTrace();
        }
        

    }

}
