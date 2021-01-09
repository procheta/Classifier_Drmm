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
public class RobertaHandling {

    Properties prop;

    public RobertaHandling(Properties prop) throws IOException {
    this.prop = prop;
    }


    public void prepareResultFile() throws IOException {

        String predictionFile = prop.getProperty("predictionFile");
	String topicFile = prop.getProperty("topicFile");
	String resultFile=prop.getProperty("resultFile");
	String qrelFile=prop.getProperty("qrelFile");    
	FileReader fr = new FileReader(new File(topicFile));
        BufferedReader br = new BufferedReader(fr);

	FileWriter fw1 = new FileWriter(new File(qrelFile));
	BufferedWriter bw1 = new BufferedWriter(new BufferedWriter(fw1));



        String line = br.readLine();
	line = br.readLine();
        HashMap<String, String> qMap = new HashMap<>();
        HashMap<String, String> qMap1 = new HashMap<>();
        //line = br.readLine();
        int count = 0;
        int count1 = 0;
        while (line != null) {
            String st[] = line.split(",");
	       if(!qMap.containsKey(st[1]))
	       {qMap.put(st[1], String.valueOf(count));
	       count++;
	       }

               if(!qMap1.containsKey(st[2])) 
	       {qMap1.put(st[2],String.valueOf(count1));
	       	count1++;
	       }
	       bw1.write(qMap.get(st[1])+" "+ qMap1.get(st[2])+" "+st[3]);
	       bw1.newLine();

            line = br.readLine();
        }
	bw1.close();
	
	fr = new FileReader(new File(predictionFile));
	br = new BufferedReader(fr);

        FileWriter fw = new FileWriter(new File(resultFile));
        BufferedWriter bw = new BufferedWriter(fw);

	
        line = br.readLine();
    	while(line != null){	
		String st[] = line.split("\t");	
            bw.write(qMap.get(st[0]) + "\t" + qMap1.get(st[1])+"\t"+st[2]);
            bw.newLine();
	    line= br.readLine();
        }
        bw.close();
    }

    

    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {

        Properties prop = new Properties();
        prop.load(new FileReader(new File("retrieve.properties")));
        RobertaHandling drp = new RobertaHandling(prop);
        drp.prepareResultFile();
    }

}
