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
public class DRMMDownSampling {
	Properties prop;

    public DRMMDownSampling(Properties prop) throws IOException {
    	this.prop = prop;
    }
    
    public void DownSampling() throws FileNotFoundException, IOException{
    
        FileReader fr = new FileReader(new File(prop.getProperty("inputFile")));
        BufferedReader br = new BufferedReader(fr);

	int nSample = Integer.parseInt(prop.getProperty("nSample"));
    
        ArrayList<String> tupleList = new ArrayList<>();
        ArrayList<String> restElements = new ArrayList<>();
        String line = br.readLine();
	line = br.readLine();
        int count = 0;
        while(line != null){
            String st[] = line.split(",");
            if(st[3].equals("1")){
		   tupleList.add(line);
            }else{
                if(count < nSample)
		{
		       	tupleList.add(line);
			count++;
		}
                else{
                    restElements.add(line);
                }
            }
            line = br.readLine();
        }  
        Collections.shuffle(tupleList);
        
        FileWriter fw = new FileWriter(new File(prop.getProperty("outputFile")));
        BufferedWriter bw = new BufferedWriter(fw);
        
        for(String s: tupleList){
            bw.write(s);
            bw.newLine();
        }
        bw.close();

	fw = new FileWriter(new File(prop.getProperty("restFile")));
	bw = new BufferedWriter(fw);
	for (String s: restElements){
		bw.write(s);
		bw.newLine();
	}
	bw.close();
    }

    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {

        Properties prop = new Properties();
        prop.load(new FileReader(new File("retrieve.properties")));
        DRMMDownSampling drp = new DRMMDownSampling(prop);
	drp.DownSampling();
    }

}
