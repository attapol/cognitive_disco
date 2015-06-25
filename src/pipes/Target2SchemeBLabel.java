package pipes;

import types.Sense;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.Alphabet;
import cc.mallet.types.Instance;
import cc.mallet.types.Label;
import cc.mallet.types.LabelAlphabet;

public class Target2SchemeBLabel extends Pipe {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public Target2SchemeBLabel() {
		// TODO Auto-generated constructor stub
	  this(null, new LabelAlphabet());
	}

	public Target2SchemeBLabel(Alphabet dataDict, Alphabet targetDict) {
		super(dataDict, targetDict);
		// TODO Auto-generated constructor stub
	}
	public Instance pipe (Instance carrier)
	{
		if (carrier.getTarget() != null) {
			if (carrier.getTarget() instanceof Label)
				throw new IllegalArgumentException ("Already a label.");
			LabelAlphabet ldict = (LabelAlphabet) getTargetAlphabet();
			String originalLabel = (String)carrier.getTarget();
			String newLabel = Sense.getSchemeBLabel(originalLabel);
			if (newLabel == null){
				carrier.setTarget(null);
			}else {
				carrier.setTarget(ldict.lookupLabel(newLabel));
			}
		}
		return carrier;
	}

}
