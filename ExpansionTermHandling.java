/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.trusteval.MSMARCO_DRMM;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Properties;

/**
 *
 * @author Procheta
 */
public class ExpansionTermHandling {

    Properties prop;

    public ExpansionTermHandling(Properties prop) throws IOException {
    this.prop = prop;
    }


    public void createQrel() throws IOException {

        String inputFile = prop.getProperty("inputFile");
	String topicFile = prop.getProperty("topicFile");
	String corpusFile=prop.getProperty("corpusFile");
	String qrelFile=prop.getProperty("qrelFile");
	String prerankedFile= prop.getProperty("prerankedFile");
	    
	FileReader fr = new FileReader(new File(inputFile));
        BufferedReader br = new BufferedReader(fr);

        FileWriter fw = new FileWriter(new File(qrelFile));
        BufferedWriter bw = new BufferedWriter(fw);
        String line = br.readLine();

        HashMap<String, Integer> qMap = new HashMap<>();
        HashMap<String, Integer> qMap1 = new HashMap<>();
        //line = br.readLine();
        int count = 0;
        int count1 = 0;
        while (line != null) {
            String st[] = line.split(",");
            if (!qMap.containsKey(st[1])) {
                qMap.put(st[1], count++);
            }
            if (!qMap1.containsKey(st[2])) {
                qMap1.put(st[2], count1++);
            }

            line = br.readLine();
        }

        fw = new FileWriter(new File(topicFile));
        bw = new BufferedWriter(fw);
        Iterator it = qMap.keySet().iterator();
        while (it.hasNext()) {
            String key = (String) it.next();
            bw.write(qMap.get(key) + "\t" + key);
            bw.newLine();
        }
        bw.close();
        fw = new FileWriter(new File(corpusFile));
        bw = new BufferedWriter(fw);
        it = qMap1.keySet().iterator();
        while (it.hasNext()) {
            String key = (String) it.next();
            bw.write(qMap1.get(key) + "\t" + key);
            bw.newLine();
        }
        bw.close();
        fw = new FileWriter(new File(qrelFile));
        bw = new BufferedWriter(fw);
        fr = new FileReader(new File(inputFile));
        br = new BufferedReader(fr);
        line = br.readLine();

        FileWriter fw1 = new FileWriter(new File(prerankedFile));
        BufferedWriter bw1 = new BufferedWriter(fw1);
        while (line != null) {
            String st[] = line.split(",");
            if (Integer.parseInt(st[3]) == 0) {
                bw.write(String.valueOf(qMap.get(st[1])) + " 0 " + String.valueOf(qMap1.get(st[2])) + " " + st[3]);
                bw.newLine();
            } else {
                // System.out.println(line);
                bw.write(String.valueOf(qMap.get(st[1])) + " 0 " + String.valueOf(qMap1.get(st[2])) + " " + st[3]);
                bw.newLine();
            }
            bw1.write(String.valueOf(qMap.get(st[1])) + "\t" + String.valueOf(qMap1.get(st[2])) + "\t0");
            bw1.newLine();
            line = br.readLine();
        }
        bw1.close();
        bw.close();
    }

    
    public String getExpansionTerms(String expansionString, int nTerms){
    
        String st[] = expansionString.split(" ");
        HashMap<Double, String> words = new HashMap<>();
        ArrayList<Double> scores = new ArrayList<>();
        for (String s: st){
         String parts[] = s.split("^");
         words.put(Double.parseDouble(parts[1]), parts[0]);
        }  
        Collections.sort(scores);
        String expandedString = "";
        for(int i = 0; i < nTerms; i++){
            double score = scores.get(i);
            expandedString += " "+ words.get(score);
        }
        return expandedString;
    }

    public void createExpandedFile() throws FileNotFoundException, IOException {
        String readFile = prop.getProperty("readFile");
	String writeFile= prop.getProperty("writeFile");
	int nTerms = Integer.parseInt(prop.getProperty("nterms"));
	FileReader fr = new FileReader(new File(readFile));
        BufferedReader br = new BufferedReader(fr);

        FileWriter fw = new FileWriter(new File(writeFile));
        BufferedWriter bw = new BufferedWriter(fw);
        String line = br.readLine();
        
        while(line != null){
            String st[] = line.split("\t");
            String expansion1 =getExpansionTerms(st[2],nTerms);
	   String expansion2 = getExpansionTerms(st[4],nTerms);

            bw.write(st[0]+"\t"+st[1]+expansion1+"\t"+st[3]+expansion2+" "+st[5]);
        	bw.newLine();
        
            line = br.readLine();
        }
	bw.close();
    }

    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {

        Properties prop = new Properties();
        prop.load(new FileReader(new File("retrieve.properties")));
        ExpansionTermHandling drp = new ExpansionTermHandling(prop);
        drp.createQrel();
        drp.createExpandedFile();
    }

}
