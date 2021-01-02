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

    public DRMMPreprocessing(Properties prop) throws IOException {
        tripleFile = prop.getProperty("DRMMtriple");
        queryFile = prop.getProperty("DRMMquery");
        File indexDir = new File(prop.getProperty("index"));
        corpusFile = prop.getProperty("corpusFile");
        qrelFile = prop.getProperty("qrels.file");
        topicFile = prop.getProperty("topicFile");
    }
    public DRMMPreprocessing(){
    
    
    }


    
    public void prepareFileForEvaluation(String querypairFile, String resultFile, String outputFile) throws FileNotFoundException, IOException{
        FileReader fr = new FileReader(new File(querypairFile));
        BufferedReader br = new BufferedReader(fr);

        FileWriter fw = new FileWriter(new File(outputFile));
        BufferedWriter bw = new BufferedWriter(fw);
        String line = br.readLine();
        String prev = line.split(" ")[0];
        HashMap<String, String> qMap = new HashMap<>();
        while (line != null) {
            String st[] = line.split(" ");
            qMap.put(st[0] + "#" + st[1], st[2]);
            line = br.readLine();
        }
        fr = new FileReader(new File(resultFile));
         br = new BufferedReader(fr);
         line = br.readLine();
         while(line != null){
            String st[] = line.split(" ");
            String key=st[0]+"#"+st[1];
            bw.write(st[2]+" "+qMap.get(key));
            bw.newLine();
             line = br.readLine();
         }  
         bw.close();
    }
    

    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {

        DRMMPreprocessing drp = new DRMMPreprocessing();
        drp.prepareFileForEvaluation(args[0],args[1],args[2]);
    }

}
