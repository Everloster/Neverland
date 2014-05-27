package org.jabe.neverland.download.log;

public class DefaultLogger implements ILogger{

	protected static final String SPLITERS = "###";
	
	protected static final String DEFAULT_TAG = "Jabe-MTD";

	@Override
	public void i(String tag, String message) {
		sysout(tag, message);
	}

	private void sysout(String tag, String message) {
		System.out.println(tag + SPLITERS + message);
	}

	@Override
	public void i(String message) {
		i(DEFAULT_TAG, message);
	}

	@Override
	public void d(String tag, String message) {
		sysout(tag, message);
	}

	@Override
	public void d(String message) {
		d(DEFAULT_TAG, message);
	}

	@Override
	public void w(String tag, String message) {
		sysout(tag, message);		
	}

	@Override
	public void w(String message) {
		w(DEFAULT_TAG, message);
	}

	@Override
	public void e(String tag, String message) {
		sysout(tag, message);
	}

	@Override
	public void e(String message) {
		e(DEFAULT_TAG, message);
	}

}
