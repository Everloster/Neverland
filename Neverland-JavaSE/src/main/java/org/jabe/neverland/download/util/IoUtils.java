package org.jabe.neverland.download.util;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public final class IoUtils {

	private static final int BUFFER_SIZE = 32 * 1024; // 32 KB

	private IoUtils() {
	}

	public static void copyStream(InputStream is, OutputStream os) throws IOException {
		byte[] bytes = new byte[BUFFER_SIZE];
		while (true) {
			int count = is.read(bytes, 0, BUFFER_SIZE);
			if (count == -1) {
				break;
			}
			os.write(bytes, 0, count);
		}
	}

	public static void closeSilently(Closeable closeable) {
		try {
			closeable.close();
		} catch (Exception e) {
			// Do nothing
		}
	}
}
