/**
 * CodedOutputStream.java
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
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.util.List;

import com.google.protobuf.ByteString;

/**
 * ClassName:CodedOutputStream <br>
 * Function: 对Nearme格式的数据进行编码，改编自google protobuf <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-15  下午2:58:19
 */
public final class CodedOutputStream {
	private final byte[] buffer;
	private final int limit;
	private int position;

	private final OutputStream output;

	/**
	 * The buffer size used in {@link #newInstance(OutputStream)}.
	 */
	public static final int DEFAULT_BUFFER_SIZE = 4096;

	/**
	 * Returns the buffer size to efficiently write dataLength bytes to this
	 * CodedOutputStream. Used by AbstractMessageLite.
	 *
	 * @return the buffer size to efficiently write dataLength bytes to this
	 *         CodedOutputStream.
	 */
	static int computePreferredBufferSize(int dataLength) {
		if (dataLength > DEFAULT_BUFFER_SIZE)
			return DEFAULT_BUFFER_SIZE;
		return dataLength;
	}

	private CodedOutputStream(final byte[] buffer, final int offset,
			final int length) {
		output = null;
		this.buffer = buffer;
		position = offset;
		limit = offset + length;
	}

	private CodedOutputStream(final OutputStream output, final byte[] buffer) {
		this.output = output;
		this.buffer = buffer;
		position = 0;
		limit = buffer.length;
	}

	/**
	 * Create a new {@code CodedOutputStream} wrapping the given
	 * {@code OutputStream}.
	 */
	public static CodedOutputStream newInstance(final OutputStream output) {
		return newInstance(output, DEFAULT_BUFFER_SIZE);
	}

	/**
	 * Create a new {@code CodedOutputStream} wrapping the given
	 * {@code OutputStream} with a given buffer size.
	 */
	public static CodedOutputStream newInstance(final OutputStream output,
			final int bufferSize) {
		return new CodedOutputStream(output, new byte[bufferSize]);
	}

	/**
	 * Create a new {@code CodedOutputStream} that writes directly to the given
	 * byte array. If more bytes are written than fit in the array,
	 * {@link OutOfSpaceException} will be thrown. Writing directly to a flat
	 * array is faster than writing to an {@code OutputStream}. See also
	 * {@link ByteString#newCodedBuilder}.
	 */
	public static CodedOutputStream newInstance(final byte[] flatArray) {
		return newInstance(flatArray, 0, flatArray.length);
	}

	/**
	 * Create a new {@code CodedOutputStream} that writes directly to the given
	 * byte array slice. If more bytes are written than fit in the slice,
	 * {@link OutOfSpaceException} will be thrown. Writing directly to a flat
	 * array is faster than writing to an {@code OutputStream}. See also
	 * {@link ByteString#newCodedBuilder}.
	 */
	public static CodedOutputStream newInstance(final byte[] flatArray,
			final int offset, final int length) {
		return new CodedOutputStream(flatArray, offset, length);
	}

	// -----------------------------------------------------------------

	/** Write a {@code double} field, including tag, to the stream. */
	public void writeDouble(final int fieldNumber, final double value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_FIXED64);
		writeDoubleNoTag(value);
	}

	/** Write a {@code float} field, including tag, to the stream. */
	public void writeFloat(final int fieldNumber, final float value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_FIXED32);
		writeFloatNoTag(value);
	}

	/** Write a {@code uint64} field, including tag, to the stream. */
	public void writeUInt64(final int fieldNumber, final long value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_VARINT);
		writeUInt64NoTag(value);
	}

	/** Write an {@code int64} field, including tag, to the stream. */
	public void writeInt64(final int fieldNumber, final long value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_VARINT);
		writeInt64NoTag(value);
	}

	/** Write an {@code int32} field, including tag, to the stream. */
	public void writeInt32(final int fieldNumber, final int value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_VARINT);
		writeInt32NoTag(value);
	}

	/** Write a {@code fixed64} field, including tag, to the stream. */
	public void writeFixed64(final int fieldNumber, final long value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_FIXED64);
		writeFixed64NoTag(value);
	}

	/** Write a {@code fixed32} field, including tag, to the stream. */
	public void writeFixed32(final int fieldNumber, final int value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_FIXED32);
		writeFixed32NoTag(value);
	}

	/** Write a {@code bool} field, including tag, to the stream. */
	public void writeBool(final int fieldNumber, final boolean value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_VARINT);
		writeBoolNoTag(value);
	}

	/** Write a {@code string} field, including tag, to the stream. */
	public void writeString(final int fieldNumber, final String value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_LENGTH_DELIMITED);
		writeStringNoTag(value);
	}

//	/** Write a {@code group} field, including tag, to the stream. */
//	public void writeGroup(final int fieldNumber, final MessageLite value)
//			throws IOException {
//		writeTag(fieldNumber, WireFormat.WIRETYPE_START_GROUP);
//		writeGroupNoTag(value);
//		writeTag(fieldNumber, WireFormat.WIRETYPE_END_GROUP);
//	}
//
	/** Write an embedded message field, including tag, to the stream. */
	public void writeMessage(final int fieldNumber, final ILazySerialize value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_LENGTH_DELIMITED);
		writeMessageNoTag(value);
	}

	/** Write an embedded message field, including tag, to the stream. */
	public void writeRepeat(final int fieldNumber, final List<?> value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_LENGTH_DELIMITED);
		writeRepeatNoTag(value);
	}

	/** Write a {@code bytes} field, including tag, to the stream. */
	public void writeBytes(final int fieldNumber, final byte[] value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_LENGTH_DELIMITED);
		writeBytesNoTag(value);
	}

	/** Write a {@code uint32} field, including tag, to the stream. */
	public void writeUInt32(final int fieldNumber, final int value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_VARINT);
		writeUInt32NoTag(value);
	}

	/**
	 * Write an enum field, including tag, to the stream. Caller is responsible
	 * for converting the enum value to its numeric value.
	 */
	public void writeEnum(final int fieldNumber, final int value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_VARINT);
		writeEnumNoTag(value);
	}

	/** Write an {@code sfixed32} field, including tag, to the stream. */
	public void writeSFixed32(final int fieldNumber, final int value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_FIXED32);
		writeSFixed32NoTag(value);
	}

	/** Write an {@code sfixed64} field, including tag, to the stream. */
	public void writeSFixed64(final int fieldNumber, final long value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_FIXED64);
		writeSFixed64NoTag(value);
	}

	/** Write an {@code sint32} field, including tag, to the stream. */
	public void writeSInt32(final int fieldNumber, final int value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_VARINT);
		writeSInt32NoTag(value);
	}

	/** Write an {@code sint64} field, including tag, to the stream. */
	public void writeSInt64(final int fieldNumber, final long value)
			throws IOException {
		writeTag(fieldNumber, WireFormat.WIRETYPE_VARINT);
		writeSInt64NoTag(value);
	}

	// -----------------------------------------------------------------

	/** Write a {@code double} field to the stream. */
	public void writeDoubleNoTag(final double value) throws IOException {
		writeRawLittleEndian64(Double.doubleToRawLongBits(value));
	}

	/** Write a {@code float} field to the stream. */
	public void writeFloatNoTag(final float value) throws IOException {
		writeRawLittleEndian32(Float.floatToRawIntBits(value));
	}

	/** Write a {@code uint64} field to the stream. */
	public void writeUInt64NoTag(final long value) throws IOException {
		writeRawVarint64(value);
	}

	/** Write an {@code int64} field to the stream. */
	public void writeInt64NoTag(final long value) throws IOException {
		writeRawVarint64(value);
	}

	/** Write an {@code int32} field to the stream. */
	public void writeInt32NoTag(final int value) throws IOException {
		if (value >= 0) {
			writeRawVarint32(value);
		} else {
			// Must sign-extend.
			writeRawVarint64(value);
		}
	}

	/** Write a {@code fixed64} field to the stream. */
	public void writeFixed64NoTag(final long value) throws IOException {
		writeRawLittleEndian64(value);
	}

	/** Write a {@code fixed32} field to the stream. */
	public void writeFixed32NoTag(final int value) throws IOException {
		writeRawLittleEndian32(value);
	}

	/** Write a {@code bool} field to the stream. */
	public void writeBoolNoTag(final boolean value) throws IOException {
		writeRawByte(value ? 1 : 0);
	}

	/** Write a {@code string} field to the stream. */
	public void writeStringNoTag(final String value) throws IOException {
		// Unfortunately there does not appear to be any way to tell Java to
		// encode
		// UTF-8 directly into our buffer, so we have to let it create its own
		// byte
		// array and then copy.
		final byte[] bytes = value.getBytes("UTF-8");
		writeRawVarint32(bytes.length);
		writeRawBytes(bytes);
	}

	/** Write a {@code bytes} field to the stream. */
	public void writeBytesNoTag(final byte[] value) throws IOException {
		writeRawVarint32(value.length);
		writeRawBytes(value);
	}

	/** Write a {@code uint32} field to the stream. */
	public void writeUInt32NoTag(final int value) throws IOException {
		writeRawVarint32(value);
	}

	/**
	 * Write an enum field to the stream. Caller is responsible for converting
	 * the enum value to its numeric value.
	 */
	public void writeEnumNoTag(final int value) throws IOException {
		writeInt32NoTag(value);
	}

	/** Write an {@code sfixed32} field to the stream. */
	public void writeSFixed32NoTag(final int value) throws IOException {
		writeRawLittleEndian32(value);
	}

	/** Write an {@code sfixed64} field to the stream. */
	public void writeSFixed64NoTag(final long value) throws IOException {
		writeRawLittleEndian64(value);
	}

	/** Write an {@code sint32} field to the stream. */
	public void writeSInt32NoTag(final int value) throws IOException {
		writeRawVarint32(encodeZigZag32(value));
	}

	/** Write an {@code sint64} field to the stream. */
	public void writeSInt64NoTag(final long value) throws IOException {
		writeRawVarint64(encodeZigZag64(value));
	}

	/** Write an embedded message field to the stream. */
	public void writeMessageNoTag(final ILazySerialize value) throws IOException {
		writeRawVarint32(value.getSerializedSize());
	    value.writeTo(this);
	}

	/** Write an embedded message field to the stream. */
	public void writeRepeatNoTag(final List<?> value) throws IOException {
		int listSize = value.size();
		writeRawVarint32(listSize);
		if (listSize > 0) {
			Object o = value.get(0);
			LazySerializeType sType = LazySerializeTypeHelper.getSerializeType(o);
			if (sType == LazySerializeType.TYPE_UNKNOWN) {
				throw new IllegalArgumentException("this type can not be handled:" + o.getClass().getName());
			}

			writeRawVarint32(sType.getType());

			for (int i = 0; i < listSize; i++) {
				writeObjectNoTag(sType, value.get(i));
			}
//
//			switch(sType) {
//			case TYPE_INT:
//				for (int i = 0; i < listSize; i++) {
//					writeInt32NoTag((Integer)value.get(i));
//				}
//
//				break;
//			case TYPE_LONG:
//				for (int i = 0; i < listSize; i++) {
//					writeInt64NoTag((Long)value.get(i));
//				}
//
//				break;
//			case TYPE_FLOAT:
//				for (int i = 0; i < listSize; i++) {
//					writeFloatNoTag((Float)value.get(i));
//				}
//
//				break;
//			case TYPE_DOUBLE:
//				for (int i = 0; i < listSize; i++) {
//					writeDoubleNoTag((Float)value.get(i));
//				}
//
//				break;
//			case TYPE_BOOLEAN:
//				for (int i = 0; i < listSize; i++) {
//					writeBoolNoTag((Boolean)value.get(i));
//				}
//
//				break;
//			case TYPE_STRING:
//				for (int i = 0; i < listSize; i++) {
//					writeStringNoTag((String)value.get(i));
//				}
//
//				break;
//			case TYPE_BYTES:
//				for (int i = 0; i < listSize; i++) {
//					writeBytesNoTag((byte[])value.get(i));
//				}
//
//				break;
//			case TYPE_MESSAGE:
//				for (int i = 0; i < listSize; i++) {
//					writeMessageNoTag((ILazySerialize)value.get(i));
//				}
//
//				break;
//			default:
//				break;
//			}
		}
	}

	public void writeObjectNoTag(LazySerializeType serializeType, Object value) throws IOException {
		switch(serializeType) {
		case TYPE_INT:
			writeInt32NoTag((Integer)value);
			break;
		case TYPE_LONG:
			writeInt64NoTag((Long)value);
			break;
		case TYPE_FLOAT:
			writeFloatNoTag((Float)value);
			break;
		case TYPE_DOUBLE:
			writeDoubleNoTag((Double)value);
			break;
		case TYPE_BOOLEAN:
			writeBoolNoTag((Boolean)value);
			break;
		case TYPE_STRING:
			writeStringNoTag((String)value);
			break;
		case TYPE_BYTES:
			writeBytesNoTag((byte[])value);
			break;
		case TYPE_MESSAGE:
			writeMessageNoTag((ILazySerialize)value);
			break;
		default:
			break;
		}
	}

	// =================================================================

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code double} field, including tag.
	 */
	public static int computeDoubleSize(final int fieldNumber,
			final double value) {
		return computeTagSize(fieldNumber) + computeDoubleSizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code float} field, including tag.
	 */
	public static int computeFloatSize(final int fieldNumber, final float value) {
		return computeTagSize(fieldNumber) + computeFloatSizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code uint64} field, including tag.
	 */
	public static int computeUInt64Size(final int fieldNumber, final long value) {
		return computeTagSize(fieldNumber) + computeUInt64SizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code int64} field, including tag.
	 */
	public static int computeInt64Size(final int fieldNumber, final long value) {
		return computeTagSize(fieldNumber) + computeInt64SizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code int32} field, including tag.
	 */
	public static int computeInt32Size(final int fieldNumber, final int value) {
		return computeTagSize(fieldNumber) + computeInt32SizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code fixed64} field, including tag.
	 */
	public static int computeFixed64Size(final int fieldNumber, final long value) {
		return computeTagSize(fieldNumber) + computeFixed64SizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code fixed32} field, including tag.
	 */
	public static int computeFixed32Size(final int fieldNumber, final int value) {
		return computeTagSize(fieldNumber) + computeFixed32SizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode a {@code bool}
	 * field, including tag.
	 */
	public static int computeBoolSize(final int fieldNumber, final boolean value) {
		return computeTagSize(fieldNumber) + computeBoolSizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code string} field, including tag.
	 */
	public static int computeStringSize(final int fieldNumber,
			final String value) {
		return computeTagSize(fieldNumber) + computeStringSizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code bytes} field, including tag.
	 */
	public static int computeBytesSize(final int fieldNumber,
			final byte[] value) {
		return computeTagSize(fieldNumber) + computeBytesSizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code uint32} field, including tag.
	 */
	public static int computeUInt32Size(final int fieldNumber, final int value) {
		return computeTagSize(fieldNumber) + computeUInt32SizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an enum field,
	 * including tag. Caller is responsible for converting the enum value to its
	 * numeric value.
	 */
	public static int computeEnumSize(final int fieldNumber, final int value) {
		return computeTagSize(fieldNumber) + computeEnumSizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code sfixed32} field, including tag.
	 */
	public static int computeSFixed32Size(final int fieldNumber, final int value) {
		return computeTagSize(fieldNumber) + computeSFixed32SizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code sfixed64} field, including tag.
	 */
	public static int computeSFixed64Size(final int fieldNumber,
			final long value) {
		return computeTagSize(fieldNumber) + computeSFixed64SizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code sint32} field, including tag.
	 */
	public static int computeSInt32Size(final int fieldNumber, final int value) {
		return computeTagSize(fieldNumber) + computeSInt32SizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code sint64} field, including tag.
	 */
	public static int computeSInt64Size(final int fieldNumber, final long value) {
		return computeTagSize(fieldNumber) + computeSInt64SizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an embedded
	 * message field, including tag.
	 */
	public static int computeMessageSize(final int fieldNumber,
			final ILazySerialize value) {
		return computeTagSize(fieldNumber) + computeMessageSizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an embedded
	 * Repeat field, including tag.
	 */
	public static int computeRepeatSize(final int fieldNumber,
			final List<?> value) {
		return computeTagSize(fieldNumber) + computeRepeatSizeNoTag(value);
	}

	// -----------------------------------------------------------------

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code double} field, including tag.
	 */
	public static int computeDoubleSizeNoTag(final double value) {
		return LITTLE_ENDIAN_64_SIZE;
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code float} field, including tag.
	 */
	public static int computeFloatSizeNoTag(final float value) {
		return LITTLE_ENDIAN_32_SIZE;
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code uint64} field, including tag.
	 */
	public static int computeUInt64SizeNoTag(final long value) {
		return computeRawVarint64Size(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code int64} field, including tag.
	 */
	public static int computeInt64SizeNoTag(final long value) {
		return computeRawVarint64Size(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code int32} field, including tag.
	 */
	public static int computeInt32SizeNoTag(final int value) {
		if (value >= 0) {
			return computeRawVarint32Size(value);
		} else {
			// Must sign-extend.
			return 10;
		}
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code fixed64} field.
	 */
	public static int computeFixed64SizeNoTag(final long value) {
		return LITTLE_ENDIAN_64_SIZE;
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code fixed32} field.
	 */
	public static int computeFixed32SizeNoTag(final int value) {
		return LITTLE_ENDIAN_32_SIZE;
	}

	/**
	 * Compute the number of bytes that would be needed to encode a {@code bool}
	 * field.
	 */
	public static int computeBoolSizeNoTag(final boolean value) {
		return 1;
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code string} field.
	 */
	public static int computeStringSizeNoTag(final String value) {
		try {
			final byte[] bytes = value.getBytes("UTF-8");
			return computeRawVarint32Size(bytes.length) + bytes.length;
		} catch (UnsupportedEncodingException e) {
			throw new RuntimeException("UTF-8 not supported.", e);
		}
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code bytes} field.
	 */
	public static int computeBytesSizeNoTag(final byte[] value) {
		return computeRawVarint32Size(value.length) + value.length;
	}

	/**
	 * Compute the number of bytes that would be needed to encode a
	 * {@code uint32} field.
	 */
	public static int computeUInt32SizeNoTag(final int value) {
		return computeRawVarint32Size(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an enum field.
	 * Caller is responsible for converting the enum value to its numeric value.
	 */
	public static int computeEnumSizeNoTag(final int value) {
		return computeInt32SizeNoTag(value);
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code sfixed32} field.
	 */
	public static int computeSFixed32SizeNoTag(final int value) {
		return LITTLE_ENDIAN_32_SIZE;
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code sfixed64} field.
	 */
	public static int computeSFixed64SizeNoTag(final long value) {
		return LITTLE_ENDIAN_64_SIZE;
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code sint32} field.
	 */
	public static int computeSInt32SizeNoTag(final int value) {
		return computeRawVarint32Size(encodeZigZag32(value));
	}

	/**
	 * Compute the number of bytes that would be needed to encode an
	 * {@code sint64} field.
	 */
	public static int computeSInt64SizeNoTag(final long value) {
		return computeRawVarint64Size(encodeZigZag64(value));
	}

	/**
	 * Compute the number of bytes that would be needed to encode an embedded
	 * message field.
	 */
	public static int computeMessageSizeNoTag(final ILazySerialize value) {
		final int size = value.getSerializedSize();
		return computeRawVarint32Size(size) + size;
	}

	/**
	 * Compute the number of bytes that would be needed to encode an embedded
	 * repeat field.
	 */
	public static int computeRepeatSizeNoTag(final List<?> value) {
		int size = 0;
		int listSize = value.size();
		if (listSize > 0) {
			Object o = value.get(0);
			LazySerializeType sType = LazySerializeTypeHelper.getSerializeType(o);
			if (sType == LazySerializeType.TYPE_UNKNOWN) {
				throw new IllegalArgumentException("this type can not be handled:" + o.getClass().getName());
			}

			//类型长度
			size += computeRawVarint32Size(sType.getType());
			for (int i = 0; i < listSize; i++) {
				size += computeValueSizeNoTag(sType, value.get(i));
			}

//			switch(sType) {
//			case TYPE_INT:
//				for (int i = 0; i < listSize; i++) {
//					size += CodedOutputStream.computeInt32SizeNoTag((Integer)value.get(i));
//				}
//				break;
//			case TYPE_LONG:
//				for (int i = 0; i < listSize; i++) {
//					size += CodedOutputStream.computeInt64SizeNoTag((Long)value.get(i));
//				}
//				break;
//			case TYPE_FLOAT:
//				for (int i = 0; i < listSize; i++) {
//					size += CodedOutputStream.computeFloatSizeNoTag((Float)value.get(i));
//				}
//				break;
//			case TYPE_DOUBLE:
//				for (int i = 0; i < listSize; i++) {
//					size += CodedOutputStream.computeDoubleSizeNoTag((Double)value.get(i));
//				}
//				break;
//			case TYPE_BOOLEAN:
//				for (int i = 0; i < listSize; i++) {
//					size += CodedOutputStream.computeBoolSizeNoTag((Boolean)value.get(i));
//				}
//				break;
//			case TYPE_STRING:
//				for (int i = 0; i < listSize; i++) {
//					size += CodedOutputStream.computeStringSizeNoTag((String)value.get(i));
//				}
//				break;
//			case TYPE_BYTES:
//				for (int i = 0; i < listSize; i++) {
//					size += CodedOutputStream.computeBytesSizeNoTag((byte[])value.get(i));
//				}
//				break;
//			case TYPE_MESSAGE:
//				for (int i = 0; i < listSize; i++) {
//					size += CodedOutputStream.computeMessageSizeNoTag((ILazySerialize)value.get(i));
//				}
//				break;
//			default:
//				break;
//			}
		}

		//列表长度+数据长度
		return computeRawVarint32Size(listSize) + size;
	}

	public static int computeValueSizeNoTag(LazySerializeType lazySerializeType, Object value) {
		switch(lazySerializeType) {
		case TYPE_INT:
			return CodedOutputStream.computeInt32SizeNoTag((Integer)value);
		case TYPE_LONG:
			return CodedOutputStream.computeInt64SizeNoTag((Long)value);
		case TYPE_FLOAT:
			return CodedOutputStream.computeFloatSizeNoTag((Float)value);
		case TYPE_DOUBLE:
			return CodedOutputStream.computeDoubleSizeNoTag((Double)value);
		case TYPE_BOOLEAN:
			return CodedOutputStream.computeBoolSizeNoTag((Boolean)value);
		case TYPE_STRING:
			return CodedOutputStream.computeStringSizeNoTag((String)value);
		case TYPE_BYTES:
			return CodedOutputStream.computeBytesSizeNoTag((byte[])value);
		case TYPE_MESSAGE:
			return CodedOutputStream.computeMessageSizeNoTag((ILazySerialize)value);
		default:
			return 0;
		}
	}

	// =================================================================

	/**
	 * Internal helper that writes the current buffer to the output. The buffer
	 * position is reset to its initial value when this returns.
	 */
	private void refreshBuffer() throws IOException {
		if (output == null) {
			// We're writing to a single buffer.
			throw new OutOfSpaceException();
		}

		// Since we have an output stream, this is our buffer
		// and buffer offset == 0
		output.write(buffer, 0, position);
		position = 0;
	}

	/**
	 * Flushes the stream and forces any buffered bytes to be written. This does
	 * not flush the underlying OutputStream.
	 */
	public void flush() throws IOException {
		if (output != null) {
			refreshBuffer();
		}
	}

	/**
	 * If writing to a flat array, return the space left in the array.
	 * Otherwise, throws {@code UnsupportedOperationException}.
	 */
	public int spaceLeft() {
		if (output == null) {
			return limit - position;
		} else {
			throw new UnsupportedOperationException(
					"spaceLeft() can only be called on CodedOutputStreams that are "
							+ "writing to a flat array.");
		}
	}

	/**
	 * Verifies that {@link #spaceLeft()} returns zero. It's common to create a
	 * byte array that is exactly big enough to hold a message, then write to it
	 * with a {@code CodedOutputStream}. Calling {@code checkNoSpaceLeft()}
	 * after writing verifies that the message was actually as big as expected,
	 * which can help catch bugs.
	 */
	public void checkNoSpaceLeft() {
		if (spaceLeft() != 0) {
			throw new IllegalStateException(
					"Did not write as much data as expected.");
		}
	}

	/**
	 * If you create a CodedOutputStream around a simple flat array, you must
	 * not attempt to write more bytes than the array has space. Otherwise, this
	 * exception will be thrown.
	 */
	public static class OutOfSpaceException extends IOException {
		private static final long serialVersionUID = -6947486886997889499L;

		OutOfSpaceException() {
			super("CodedOutputStream was writing to a flat byte array and ran "
					+ "out of space.");
		}
	}

	/** Write a single byte. */
	public void writeRawByte(final byte value) throws IOException {
		if (position == limit) {
			refreshBuffer();
		}

		buffer[position++] = value;
	}

	/** Write a single byte, represented by an integer value. */
	public void writeRawByte(final int value) throws IOException {
		writeRawByte((byte) value);
	}

	/** Write an array of bytes. */
	public void writeRawBytes(final byte[] value) throws IOException {
		writeRawBytes(value, 0, value.length);
	}

	/** Write part of an array of bytes. */
	public void writeRawBytes(final byte[] value, int offset, int length)
			throws IOException {
		if (limit - position >= length) {
			// We have room in the current buffer.
			System.arraycopy(value, offset, buffer, position, length);
			position += length;
		} else {
			// Write extends past current buffer. Fill the rest of this buffer
			// and
			// flush.
			final int bytesWritten = limit - position;
			System.arraycopy(value, offset, buffer, position, bytesWritten);
			offset += bytesWritten;
			length -= bytesWritten;
			position = limit;
			refreshBuffer();

			// Now deal with the rest.
			// Since we have an output stream, this is our buffer
			// and buffer offset == 0
			if (length <= limit) {
				// Fits in new buffer.
				System.arraycopy(value, offset, buffer, 0, length);
				position = length;
			} else {
				// Write is very big. Let's do it all at once.
				output.write(value, offset, length);
			}
		}
	}

	/** Encode and write a tag. */
	public void writeTag(final int fieldNumber, final int wireType)
			throws IOException {
		writeRawVarint32(WireFormat.makeTag(fieldNumber, wireType));
	}

	/** Compute the number of bytes that would be needed to encode a tag. */
	public static int computeTagSize(final int fieldNumber) {
		return computeRawVarint32Size(WireFormat.makeTag(fieldNumber, 0));
	}

	/**
	 * Encode and write a varint. {@code value} is treated as unsigned, so it
	 * won't be sign-extended if negative.
	 */
	public void writeRawVarint32(int value) throws IOException {
		while (true) {
			if ((value & ~0x7F) == 0) {
				writeRawByte(value);
				return;
			} else {
				writeRawByte((value & 0x7F) | 0x80);
				value >>>= 7;
			}
		}
	}

	/**
	 * Compute the number of bytes that would be needed to encode a varint.
	 * {@code value} is treated as unsigned, so it won't be sign-extended if
	 * negative.
	 */
	public static int computeRawVarint32Size(final int value) {
		if ((value & (0xffffffff << 7)) == 0)
			return 1;
		if ((value & (0xffffffff << 14)) == 0)
			return 2;
		if ((value & (0xffffffff << 21)) == 0)
			return 3;
		if ((value & (0xffffffff << 28)) == 0)
			return 4;
		return 5;
	}

	/** Encode and write a varint. */
	public void writeRawVarint64(long value) throws IOException {
		while (true) {
			if ((value & ~0x7FL) == 0) {
				writeRawByte((int) value);
				return;
			} else {
				writeRawByte(((int) value & 0x7F) | 0x80);
				value >>>= 7;
			}
		}
	}

	/** Compute the number of bytes that would be needed to encode a varint. */
	public static int computeRawVarint64Size(final long value) {
		if ((value & (0xffffffffffffffffL << 7)) == 0)
			return 1;
		if ((value & (0xffffffffffffffffL << 14)) == 0)
			return 2;
		if ((value & (0xffffffffffffffffL << 21)) == 0)
			return 3;
		if ((value & (0xffffffffffffffffL << 28)) == 0)
			return 4;
		if ((value & (0xffffffffffffffffL << 35)) == 0)
			return 5;
		if ((value & (0xffffffffffffffffL << 42)) == 0)
			return 6;
		if ((value & (0xffffffffffffffffL << 49)) == 0)
			return 7;
		if ((value & (0xffffffffffffffffL << 56)) == 0)
			return 8;
		if ((value & (0xffffffffffffffffL << 63)) == 0)
			return 9;
		return 10;
	}

	/** Write a little-endian 32-bit integer. */
	public void writeRawLittleEndian32(final int value) throws IOException {
		writeRawByte((value) & 0xFF);
		writeRawByte((value >> 8) & 0xFF);
		writeRawByte((value >> 16) & 0xFF);
		writeRawByte((value >> 24) & 0xFF);
	}

	public static final int LITTLE_ENDIAN_32_SIZE = 4;

	/** Write a little-endian 64-bit integer. */
	public void writeRawLittleEndian64(final long value) throws IOException {
		writeRawByte((int) (value) & 0xFF);
		writeRawByte((int) (value >> 8) & 0xFF);
		writeRawByte((int) (value >> 16) & 0xFF);
		writeRawByte((int) (value >> 24) & 0xFF);
		writeRawByte((int) (value >> 32) & 0xFF);
		writeRawByte((int) (value >> 40) & 0xFF);
		writeRawByte((int) (value >> 48) & 0xFF);
		writeRawByte((int) (value >> 56) & 0xFF);
	}

	public static final int LITTLE_ENDIAN_64_SIZE = 8;

	/**
	 * Encode a ZigZag-encoded 32-bit value. ZigZag encodes signed integers into
	 * values that can be efficiently encoded with varint. (Otherwise, negative
	 * values must be sign-extended to 64 bits to be varint encoded, thus always
	 * taking 10 bytes on the wire.)
	 *
	 * @param n
	 *            A signed 32-bit integer.
	 * @return An unsigned 32-bit integer, stored in a signed int because Java
	 *         has no explicit unsigned support.
	 */
	public static int encodeZigZag32(final int n) {
		// Note: the right-shift must be arithmetic
		return (n << 1) ^ (n >> 31);
	}

	/**
	 * Encode a ZigZag-encoded 64-bit value. ZigZag encodes signed integers into
	 * values that can be efficiently encoded with varint. (Otherwise, negative
	 * values must be sign-extended to 64 bits to be varint encoded, thus always
	 * taking 10 bytes on the wire.)
	 *
	 * @param n
	 *            A signed 64-bit integer.
	 * @return An unsigned 64-bit integer, stored in a signed int because Java
	 *         has no explicit unsigned support.
	 */
	public static long encodeZigZag64(final long n) {
		// Note: the right-shift must be arithmetic
		return (n << 1) ^ (n >> 63);
	}
}

