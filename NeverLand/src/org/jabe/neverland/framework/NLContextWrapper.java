package org.jabe.neverland.framework;

public class NLContextWrapper extends NLContext {
	
	private NLContext mBaseContext;
	
	public NLContextWrapper(NLContext context) {
		this.mBaseContext = context;
	}

	@Override
	public Object getSystemService(String name) {
		return mBaseContext.getSystemService(name);
	}

}
