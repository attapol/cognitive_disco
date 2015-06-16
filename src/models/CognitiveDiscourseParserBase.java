package models;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import org.json.JSONException;

import types.DataTriplet;
import types.DimensionMapper;

public abstract class CognitiveDiscourseParserBase {

	protected DimensionMapper dm;
	protected DataTriplet[] data;
	
	public CognitiveDiscourseParserBase(String experimentName, String mappingJson, String trainingDir, String devDir, String testDir) throws JSONException, IOException{
		dm = new DimensionMapper(mappingJson);
		data = new DataTriplet[dm.numDimensions()];
		ArrayList<String> dimensions = dm.getDimensions();
		for (int i = 0; i < dimensions.size(); i++){
			String dimension = dimensions.get(i);
			data[i] = loadData(experimentName, dimension, trainingDir, devDir, testDir);
		}
	}
	
	private DataTriplet loadData(String experimentName, String dimension, String trainingDir, String devDir, String testDir) throws FileNotFoundException{
		String trainingFileName = dm.getFeatureFileName(experimentName, dimension, trainingDir);
		System.out.println("Reading dev set");
		String devFileName = dm.getFeatureFileName(experimentName, dimension, devDir);
		System.out.println("Reading test set");
		String testFileName = dm.getFeatureFileName(experimentName, dimension, testDir);
		return new DataTriplet(trainingFileName, devFileName, testFileName);
	}
	
	abstract public void trainTest();
	
	/*
	 * Supposedly, it will call the Python featurizer and write features to file.
	 * Then classify the data 
	 */
	abstract public void classify(String dataDir);
}
