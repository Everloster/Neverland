package org.jabe.neverland.download.log;

public interface ILogger {
	public void i(String tag, String message);
	public void i(String message);
	public void d(String tag, String message);
	public void d(String message);
	public void w(String tag, String message);
	public void w(String message);
	public void e(String tag, String message);
	public void e(String message);
}
