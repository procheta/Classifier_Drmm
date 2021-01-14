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
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Properties;

/**
 *
 * @author Procheta
 */
public class DRMMPreprocessing {

    String tripleFile;
    String corpusFile;
    String queryFile;
    HashMap<String, String> qidMap;
    String topicFile;
    String qrelFile;

    public DRMMPreprocessing(){
    
    
    }


    
    public void prepareFileForEvaluation(String querypairFile,String querypairFile2, String resultFile, String outputFile) throws FileNotFoundException, IOException{
        FileReader fr = new FileReader(new File(querypairFile));
        BufferedReader br = new BufferedReader(fr);

        FileWriter fw = new FileWriter(new File(outputFile));
        BufferedWriter bw = new BufferedWriter(fw);
        String line = br.readLine();
        HashMap<String, String> qMap = new HashMap<>();
        while (line != null) {
            String st[] = line.split("\t");
            qMap.put(st[1].trim() , st[0]);
            line = br.readLine();
        }

	fr = new FileReader(new File(querypairFile2));
	br = new BufferedReader(new BufferedReader(fr));
	line = br.readLine();
	HashMap<String,String> qMap1 = new HashMap<>();
	while(line != null){
		String st[] = line.split("\t");
		qMap1.put(st[1].trim(),st[0]);
		line = br.readLine();
	}
        fr = new FileReader(new File(resultFile));
         br = new BufferedReader(fr);
         line = br.readLine();
         while(line != null){
            String st[] = line.split("\t");
	    //System.out.println("hh"+st[0].trim());
            bw.write(qMap.get(st[0].trim())+" "+qMap1.get(st[1].trim())+" "+st[2]);
            bw.newLine();
            line = br.readLine();
         }  
         bw.close();
    }
    

    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {

        DRMMPreprocessing drp = new DRMMPreprocessing();
        drp.prepareFileForEvaluation(args[0],args[1],args[2],args[3]);
    }

}
