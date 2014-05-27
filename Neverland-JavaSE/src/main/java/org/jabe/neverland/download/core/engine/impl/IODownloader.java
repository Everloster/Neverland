package org.jabe.neverland.download.core.engine.impl;

import java.io.IOException;
import java.io.InputStream;

public interface IODownloader {

	InputStream getStream(String imageUri, Object extra, SizeBean sb)
			throws IOException;

	/**
	 * Represents supported schemes(protocols) of URI. Provides convenient
	 * methods for work with schemes and URIs.
	 */
	public enum Scheme {
		HTTP("http"), HTTPS("https"), FILE("file"), CONTENT("content"), ASSETS(
				"assets"), DRAWABLE("drawable"), UNKNOWN("");

		private String scheme;
		private String uriPrefix;

		Scheme(String scheme) {
			this.scheme = scheme;
			uriPrefix = scheme + "://";
		}

		/**
		 * Defines scheme of incoming URI
		 * 
		 * @param uri
		 *            URI for scheme detection
		 * @return Scheme of incoming URI
		 */
		public static Scheme ofUri(String uri) {
			if (uri != null) {
				for (Scheme s : values()) {
					if (s.belongsTo(uri)) {
						return s;
					}
				}
			}
			return UNKNOWN;
		}

		private boolean belongsTo(String uri) {
			return uri.startsWith(uriPrefix);
		}

		/** Appends scheme to incoming path */
		public String wrap(String path) {
			return uriPrefix + path;
		}

		/** Removed scheme part ("scheme://") from incoming URI */
		public String crop(String uri) {
			if (!belongsTo(uri)) {
				throw new IllegalArgumentException(String.format(
						"URI [%1$s] doesn't have expected scheme [%2$s]", uri,
						scheme));
			}
			return uri.substring(uriPrefix.length());
		}
	}

	public static class SizeBean {

		public long contentLength;
		public long start;
		public long end;

		public SizeBean(long contentLength, long start, long end) {
			super();
			this.contentLength = contentLength;
			this.start = start;
			this.end = end;
		}
	}
}
