/**
 * LazySerializeException.java
 * com.nearme.base.serializer
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-15 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.serializer;

import java.io.IOException;

/**
 * ClassName:LazySerializeException <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-15  下午3:14:15
 */
public class LazySerializeException extends IOException {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;

	public LazySerializeException() {
	}

	public LazySerializeException(String msg) {
		super(msg);
	}

	static LazySerializeException truncatedMessage() {
		return new LazySerializeException(
				"While parsing a LazySerialize message, the input ended unexpectedly "
						+ "in the middle of a field.  This could mean either than the "
						+ "input has been truncated or that an embedded message "
						+ "misreported its own length.");
	}

	static LazySerializeException negativeSize() {
		return new LazySerializeException(
				"CodedInputStream encountered an embedded string or message "
						+ "which claimed to have negative size.");
	}

	static LazySerializeException malformedVarint() {
		return new LazySerializeException(
				"CodedInputStream encountered a malformed varint.");
	}

	static LazySerializeException invalidTag() {
		return new LazySerializeException(
				"LazySerialize message contained an invalid tag (zero).");
	}

	static LazySerializeException invalidEndTag() {
		return new LazySerializeException(
				"LazySerialize message end-group tag did not match expected tag.");
	}

	static LazySerializeException invalidWireType() {
		return new LazySerializeException(
				"LazySerialize message tag had invalid wire type.");
	}

	static LazySerializeException recursionLimitExceeded() {
		return new LazySerializeException(
				"LazySerialize message had too many levels of nesting.  May be malicious.  "
						+ "Use CodedInputStream.setRecursionLimit() to increase the depth limit.");
	}

	static LazySerializeException sizeLimitExceeded() {
		return new LazySerializeException(
				"NearmeSerialize message was too large.  May be malicious.  "
						+ "Use CodedInputStream.setSizeLimit() to increase the size limit.");
	}
}

