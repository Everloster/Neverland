package org.jabe.neverland.pattern;

public class ChainOfResponsibility {

	public interface RequestHandle {

		public void handleRequest(Request request);

	}

	public interface Request {
		public String getRequestContent();
	}

	public static class HttpHandle implements RequestHandle {

		@Override
		public void handleRequest(Request request) {
			System.out.println("HttpHandle handle the message : "
					+ request.getRequestContent());
		}

	}

	public static class FTPHandle implements RequestHandle {

		@Override
		public void handleRequest(Request request) {
			System.out.println("FTPHandle handle the message : "
					+ request.getRequestContent());
		}

	}

	public static class HttpRequest implements Request {

		@Override
		public String getRequestContent() {
			// TODO Auto-generated method stub
			return "Http: message";
		}

	}

	public static class FTPRequeset implements Request {

		@Override
		public String getRequestContent() {
			return "FTP: message";
		}

	}

	public static void main(String[] args) {
		
		final Request request1 = new FTPRequeset();
		final Request request2 = new HttpRequest();
		final ChainOfResponsibility app = new ChainOfResponsibility();
		app.handleRequest(request1);
		app.handleRequest(request2);
		
	}

	public void handleRequest(Request request) {
		
		if (request.getRequestContent().startsWith("Http")
				|| request instanceof HttpRequest) {
			new HttpHandle().handleRequest(request);
		}

		if (request.getRequestContent().startsWith("FTP")
				|| request instanceof FTPRequeset) {
			new FTPHandle().handleRequest(request);
		}
		
	}
}
