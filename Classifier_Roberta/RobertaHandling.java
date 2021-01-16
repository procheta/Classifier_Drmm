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

	FileWriter fw = new FileWriter(new File(qrelFile));
	BufferedWriter bw = new BufferedWriter(new BufferedWriter(fw));



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

            line = br.readLine();
        }
	


	Iterator it = qMap.keySet().iterator();
    	while(it.hasNext()){	
		String st =(String)it.next();	
            bw.write(st+"\t"+qMap.get(st) );
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
