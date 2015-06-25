package types;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.json.JSONException;
import org.json.JSONObject;

public class DimensionMapper {

	private JSONObject mapping;
	private ArrayList<String> dimensions = new ArrayList<String>();
	private ArrayList<Sense> senses = new ArrayList<Sense>();
	private String mappingName;
	private HashMap<ArrayList<String>, HashSet<Sense> > dimensionsToSense = new HashMap<ArrayList<String>, HashSet<Sense>>();
	
	
	public DimensionMapper() {
		
	}
	
	public DimensionMapper(String mappingJson) throws IOException, JSONException{
		String jsonString = FileUtils.readFileToString(new File(mappingJson));
		mapping = new JSONObject(jsonString);
		mappingName = FilenameUtils.getBaseName(mappingJson);
		
		// collect all senses
		Iterator<?> it = mapping.keys();
		while (it.hasNext()) senses.add(new Sense((String)it.next()));
		
		// collect all dimension
		it = mapping.keys();
		String key = (String)it.next();
		it = mapping.getJSONObject(key).keys();
		while (it.hasNext()) dimensions.add((String)it.next());
		
		for (Sense sense : senses){
			ArrayList<String> labelDimensions = new ArrayList<String>();
			for (int i = 0; i < numDimensions(); i++){
				String dimension = dimensions.get(i);
				labelDimensions.add(mapping.getJSONObject(sense.getRawSense()).getString(dimension));
			}
			if (dimensionsToSense.containsKey(labelDimensions)){
				dimensionsToSense.get(labelDimensions).add(sense);
			} else {
				HashSet<Sense> newSenseSet = new HashSet<Sense>();
				newSenseSet.add(sense);
				dimensionsToSense.put(labelDimensions, newSenseSet);
			}
		}
		
	}
	
	public String getLabel(ArrayList<String> dimensions, LabelType labelType) {
		HashSet<Sense> senses = this.dimensionsToSense.get(dimensions);
		HashSet<String> labelSet = new HashSet<String>();
		for (Sense sense : senses){
			if (sense.isFinestSense()) labelSet.add(sense.getLabel(labelType));
		}
		Object[] labels = labelSet.toArray();
		if (labels.length == 0) {
			switch (labelType) {
			case TOP_LEVEL:
				return "Expansion";
			case CONLL:
				return "Expansion.Conjunction";
			case SCHEME_B:
				return "Expansion.Conjunction";
			}
			
		} else if (labels.length != 1) {
			return (String)labels[0];
		} else {
			return (String)labels[0];
		}
		return null;
		
	}


	public int numDimensions() {
		return dimensions.size();
	}
	
	public ArrayList<String> getDimensions() {
		return dimensions;
	}

	public String getFeatureFileName(String experimentName, String dimension, String trainingDir) {
		return trainingDir + "/" + experimentName + "." + mappingName + "." + dimension + ".features";
	}
	
	public String getFeatureFileName(String experimentName, String trainingDir) {
		return trainingDir + "/" + experimentName + ".original_label.features";
	}
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(mappingName + "\n\n");

		sb.append("=============================================\n");
		sb.append("All levels\n");
		for (ArrayList<String> dimension : dimensionsToSense.keySet()){
			HashSet<Sense> senses = dimensionsToSense.get(dimension);
			sb.append(String.join(" + ", dimension));
			//for (String d : dimension) sb.append(d + " + ");
			sb.append(" -->\n");
			for (Sense sense : senses){
				sb.append("\t" + sense.getRawSense() + "\n");
			}
			sb.append("\n");
		}
		
		sb.append("=============================================\n");
		sb.append("Top level only\n");
		for (ArrayList<String> dimension : dimensionsToSense.keySet()){
			HashSet<Sense> senses = dimensionsToSense.get(dimension);
			for (String d : dimension) sb.append(d + " + ");
			sb.append(" -->\n");
			
            HashSet<String> seenLabels = new HashSet<String>();
			for (Sense sense : senses){
				String label = sense.getTopLevelLabel();
				if (!seenLabels.contains(label)){
					sb.append("\t" + label + "\n");
					seenLabels.add(label);
				}
			}
			sb.append("\n");
		}

		sb.append("=============================================\n");
		sb.append("Modified level 2 only\n");
		for (ArrayList<String> dimension : dimensionsToSense.keySet()){
			HashSet<Sense> senses = dimensionsToSense.get(dimension);
			for (String d : dimension) sb.append(d + " + ");
			sb.append(" -->\n");
			
            HashSet<String> seenLabels = new HashSet<String>();
			for (Sense sense : senses){
				if (sense.isFinestSense()) {
					String label = sense.getSchemeBLabel();
					if (!seenLabels.contains(label) && !label.equalsIgnoreCase(Sense.NULL_SENSE)){
						sb.append("\t" + label + "\n");
						seenLabels.add(label);
					}
				}
			}
			sb.append("\n");
		}

		sb.append("=============================================\n");
		sb.append("CoNLL Shared Task Labels only\n");
		for (ArrayList<String> dimension : dimensionsToSense.keySet()){
			HashSet<Sense> senses = dimensionsToSense.get(dimension);
			for (String d : dimension) sb.append(d + " + ");
			sb.append(" -->\n");
			
            HashSet<String> seenLabels = new HashSet<String>();
			for (Sense sense : senses){
				if (sense.isFinestSense()) {
					String label = sense.getCoNLLLabel();
					if (!seenLabels.contains(label) && !label.equals(Sense.NULL_SENSE)){
						sb.append("\t" + label + "\n");
						seenLabels.add(label);
					}
				}
			}
			sb.append("\n");
		}
		
		return sb.toString();
		
	}

	public static void main(String[] args) throws JSONException, IOException {
		//DimensionMapper dm = new DimensionMapper(args[0]);
		DimensionMapper dm = new DimensionMapper("mapping3b.json");
		System.out.println(dm.toString());
	}
}
