package types;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

import org.json.JSONException;
import org.json.JSONObject;

public class ResultWriter {

	private PrintWriter writer;
	private int numResults;
	
	public ResultWriter(String fileName) throws IOException{
		writer = new PrintWriter(new File(fileName));
	}
	
	public ResultWriter(OutputStream out) {
		writer = new PrintWriter(out);
	}
	
	public void write(SimpleConfusionMatrix cm) throws JSONException {
		write(cm, new HashMap<String, String>());
	}
	
	public void write(SimpleConfusionMatrix cm, Map<String, String> extraInformation) throws JSONException{
		JSONObject json = cm.toJson();
		for (String key: extraInformation.keySet()){
			json.put(key, extraInformation.get(key));
		}
		if (numResults > 0) writer.write("\n");
		writer.write(json.toString());
		numResults++;
	}

}
