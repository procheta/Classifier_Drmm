/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

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
public class ExpansionTermsHandling{


    public ExpansionTermsHandling() throws IOException {
    }

    
    public String getExpansionTerms(String expansionString, int nTerms){
    
        String st[] = expansionString.split(" ");
        HashMap<Double, String> words = new HashMap<>();
        ArrayList<Double> scores = new ArrayList<>();
        for (String s: st){
	       	s= s.replace("^", " ");
		String parts[] = s.split(" ");
         words.put(Double.parseDouble(parts[1]), parts[0]);
	 scores.add(Double.parseDouble(parts[1]));
        }  
        Collections.sort(scores);
        String expandedString = "";
        for(int i = 0; i < nTerms; i++){
            double score = scores.get(i);
            expandedString += " "+ words.get(score);
        }
        return expandedString;
    }

    public void createExpandedFile(String readFile, String writeFile, int nTerms) throws FileNotFoundException, IOException {
        FileReader fr = new FileReader(new File(readFile));
        BufferedReader br = new BufferedReader(fr);

        FileWriter fw = new FileWriter(new File(writeFile));
        BufferedWriter bw = new BufferedWriter(fw);
        String line = br.readLine();
        
        while(line != null){
            String st[] = line.split("\t");
            String expansioTerms = st[2];
            String expansion1=getExpansionTerms(st[2],nTerms);
	    String expansion2 = getExpansionTerms(st[4],nTerms);
            bw.write(st[0]+"\t"+st[1]+expansion1+"\t"+st[3]+expansion2+"\t"+st[5]);
	    bw.newLine();
            line = br.readLine();
        }
	bw.close();
    }

    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {

        Properties prop = new Properties();
        ExpansionTermsHandling drp = new ExpansionTermsHandling();
        drp.createExpandedFile(args[0], args[1],Integer.parseInt(args[2]));
    }

}
