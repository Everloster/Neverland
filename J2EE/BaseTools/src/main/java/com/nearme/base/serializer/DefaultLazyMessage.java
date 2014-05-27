/**
 * DefaultLazyMessage.java
 * com.nearme.base.serializer
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-5-16 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.serializer;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * ClassName:DefaultLazyMessage <br>
 * Function: TODO ADD FUNCTION <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-5-16  上午10:26:13
 */
public class DefaultLazyMessage extends AbstractLazySerialize implements ILazyMessage {
	private static final long serialVersionUID = 1L;

	private Map<Integer, Object> valueMap;	//key为列索引
	private ILazyMetaData metaData;
	private boolean isMetaDataFromOutside;	//元数据是否是来自外部，如果是则在输出时不用输出

	public DefaultLazyMessage() {
		super();

		valueMap = new HashMap<Integer, Object>();
	}

	/**
	 *
	 * 创建一个指定元数据的序列化消息.
	 *
	 * @param metaData
	 */
	public DefaultLazyMessage(ILazyMetaData metaData) {
		this();

		this.setMetaData(metaData);
	}

	@Override
	public void setMetaData(ILazyMetaData metaData) {
		if (null != this.metaData) {
			throw new IllegalArgumentException("meta data is setting, you can't set it twice!");
		}

		if (null != metaData) {
			this.metaData = metaData;
			isMetaDataFromOutside = true;
		} else {
			isMetaDataFromOutside = false;
		}
	}

	@Override
	public ILazyMetaData getMetaData() {
		return this.metaData;
	}

	@Override
	public boolean isMetaDataFromOutside() {
		return isMetaDataFromOutside;
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#setInt(int, int)
	 */
	@Override
	public void setInt(int columnIndex, int value) {
		checkColumnIndex(columnIndex);

		setColumnType(columnIndex, LazySerializeType.TYPE_INT);
		valueMap.put(columnIndex, value);

		resetSerializeSize();
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#getInt(int)
	 */
	@Override
	public int getInt(int columnIndex) {
		Integer value = (Integer)valueMap.get(columnIndex);
		return null == value ? 0 : value;
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#setLong(int, long)
	 */
	@Override
	public void setLong(int columnIndex, long value) {
		checkColumnIndex(columnIndex);

		setColumnType(columnIndex, LazySerializeType.TYPE_LONG);
		valueMap.put(columnIndex, value);

		resetSerializeSize();
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#getLong(int)
	 */
	@Override
	public long getLong(int columnIndex) {
		Long value = (Long)valueMap.get(columnIndex);
		return null == value ? 0 : value;
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#setFloat(int, float)
	 */
	@Override
	public void setFloat(int columnIndex, float value) {
		checkColumnIndex(columnIndex);

		setColumnType(columnIndex, LazySerializeType.TYPE_FLOAT);
		valueMap.put(columnIndex, value);

		resetSerializeSize();
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#getFloat(int)
	 */
	@Override
	public float getFloat(int columnIndex) {
		Float value = (Float)valueMap.get(columnIndex);
		return null == value ? 0 : value;
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#setDouble(int, double)
	 */
	@Override
	public void setDouble(int columnIndex, double value) {
		checkColumnIndex(columnIndex);

		setColumnType(columnIndex, LazySerializeType.TYPE_DOUBLE);
		valueMap.put(columnIndex, value);

		resetSerializeSize();
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#getDouble(int)
	 */
	@Override
	public double getDouble(int columnIndex) {
		Double value = (Double)valueMap.get(columnIndex);
		return null == value ? 0 : value;
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#setBoolean(int, boolean)
	 */
	@Override
	public void setBoolean(int columnIndex, boolean value) {
		checkColumnIndex(columnIndex);

		setColumnType(columnIndex, LazySerializeType.TYPE_BOOLEAN);
		valueMap.put(columnIndex, value);

		resetSerializeSize();
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#getBoolean(int)
	 */
	@Override
	public boolean getBoolean(int columnIndex) {
		Boolean value = (Boolean)valueMap.get(columnIndex);
		return null == value ? false : value;
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#setString(int, java.lang.String)
	 */
	@Override
	public void setString(int columnIndex, String value) {
		checkColumnIndex(columnIndex);

		setColumnType(columnIndex, LazySerializeType.TYPE_STRING);

		if (null == value) {
			return;
		}
		valueMap.put(columnIndex, value);

		resetSerializeSize();
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#getString(int)
	 */
	@Override
	public String getString(int columnIndex) {
		return (String)valueMap.get(columnIndex);
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#setBytes(int, byte[])
	 */
	@Override
	public void setBytes(int columnIndex, byte[] value) {
		checkColumnIndex(columnIndex);

		setColumnType(columnIndex, LazySerializeType.TYPE_BYTES);

		if (null == value) {
			return;
		}
		valueMap.put(columnIndex, value);

		resetSerializeSize();
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#getBytes(int)
	 */
	@Override
	public byte[] getBytes(int columnIndex) {
		return (byte[])valueMap.get(columnIndex);
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#setMessage(int, com.nearme.base.serializer.ILazySerialize)
	 */
	@Override
	public void setMessage(int columnIndex, ILazySerialize value) {
		checkColumnIndex(columnIndex);

		setColumnType(columnIndex, LazySerializeType.TYPE_MESSAGE);

		if (null == value) {
			return;
		}
		valueMap.put(columnIndex, value);

		resetSerializeSize();
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#getMessage(int)
	 */
	@Override
	public ILazyMessage getMessage(int columnIndex) {
		return (ILazyMessage)valueMap.get(columnIndex);
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#setList(int, java.util.List)
	 */
	@Override
	public void setList(int columnIndex, List<?> value) {
		checkColumnIndex(columnIndex);

		setColumnType(columnIndex, LazySerializeType.TYPE_REPEAT);

		if (null == value) {
			valueMap.remove(columnIndex);
		} else {
			valueMap.put(columnIndex, value);
		}

		resetSerializeSize();
	}

	/**
	 * @see com.nearme.base.serializer.ILazyMessage#getList(int)
	 */
	@Override
	public List<?> getList(int columnIndex) {
		return (List<?>)valueMap.get(columnIndex);
	}

	private void setColumnType(int columnIndex, LazySerializeType serializeType) {
		if (null == metaData) {
			metaData = LazySerializeFactory.createLazyMetaData();
		}

		metaData.setColumnType(columnIndex, serializeType);
	}

	@Override
	public void addIntToList(int columnIndex, int value) {
		addToList(columnIndex, value);
	}

	@Override
	public void addLongToList(int columnIndex, long value) {
		addToList(columnIndex, value);
	}

	@Override
	public void addFloatToList(int columnIndex, float value) {
		addToList(columnIndex, value);
	}

	@Override
	public void addDoubleToList(int columnIndex, double value) {
		addToList(columnIndex, value);
	}

	@Override
	public void addBooleanToList(int columnIndex, boolean value) {
		addToList(columnIndex, value);
	}

	@Override
	public void addStringToList(int columnIndex, String value) {
		addToList(columnIndex, value);
	}

	@Override
	public void addBytesToList(int columnIndex, byte[] value) {
		addToList(columnIndex, value);
	}

	@Override
	public void addMessageToList(int columnIndex, ILazySerialize value) {
		addToList(columnIndex, value);
	}

	@Override
	public void addAllToList(int columnIndex, List<?> value) {
		checkColumnIndex(columnIndex);

		setColumnType(columnIndex, LazySerializeType.TYPE_REPEAT);

		if (null == value) {
			return;
		}

		Object oldValue = valueMap.get(columnIndex);
		if (null == oldValue) {
			valueMap.put(columnIndex, value);
		} else if (oldValue instanceof List) {
			@SuppressWarnings("unchecked")
			List<Object> list = (List<Object>)oldValue;
			list.addAll(value);
		} else {
			throw new IllegalArgumentException("expect type List, but get " + oldValue.getClass().getName());
		}

		resetSerializeSize();
	}

	/**
	 * 此方法只能添加SerializeType中除TYPE_REPEAT的类型
	 * @param
	 * @return
	 */
	@SuppressWarnings("unchecked")
	private void addToList(int columnIndex, Object value) {
		checkColumnIndex(columnIndex);

		setColumnType(columnIndex, LazySerializeType.TYPE_REPEAT);

		if (null == value) {
			return;
		}

		Object oldValue = valueMap.get(columnIndex);

		if (null == oldValue) {
			List<Object> list = new ArrayList<Object>();
			list.add(value);
			valueMap.put(columnIndex, list);
		} else if (oldValue instanceof List) {
			List<Object> list = (List<Object>)oldValue;
			list.add(value);
		} else {
			throw new IllegalArgumentException("expect type List, but get " + oldValue.getClass().getName());
		}

		resetSerializeSize();
	}

	@Override
	public void writeTo(CodedOutputStream output) throws IOException {
		//如果元数据不是外部传入的，则需要输出,同时需要输出协议的长度及版本
		if (!isMetaDataFromOutside) {
			output.writeInt32NoTag(this.getSerializedSize());
			output.writeInt32NoTag(this.getCurrentVersion());

			output.writeMessageNoTag(metaData);
			//metaData.writeTo(output);
		}

		//写入元数据
		for (Integer key : valueMap.keySet()) {
			LazySerializeType lazySerializeType = metaData.getColumnType(key);
			if (LazySerializeType.TYPE_UNKNOWN == lazySerializeType) {
				throw new RuntimeException("column index not fount:" + key);
			}

			Object value = valueMap.get(key);
			switch (lazySerializeType) {
			case TYPE_INT:
				output.writeInt32(key, (Integer)value);
				break;
			case TYPE_LONG:
				output.writeInt64(key, (Long)value);
				break;
			case TYPE_FLOAT:
				output.writeFloat(key, (Float)value);
				break;
			case TYPE_DOUBLE:
				output.writeDouble(key, (Double)value);
				break;
			case TYPE_BOOLEAN:
				output.writeBool(key, (Boolean)value);
				break;
			case TYPE_STRING:
				output.writeString(key, (String)value);
				break;
			case TYPE_BYTES:
				output.writeBytes(key, (byte[])value);
				break;
			case TYPE_MESSAGE:
				output.writeMessage(key, (ILazySerialize)value);
				break;
			case TYPE_REPEAT:
				output.writeRepeat(key, (List<?>)value);
				break;
			default:
				//忽略不识别的类型
				continue;
			}
		}
	}

	@Override
	protected int computeSerializeSize() {
		if (null == metaData) {
			return 0;
		}

		int size = 0;

		//如果元数据不是外部传入的，则需要输出
		if (!isMetaDataFromOutside) {
			int metaDataSize = CodedOutputStream.computeMessageSizeNoTag(metaData);
			if (metaDataSize == 0) {
				return 0;
			}

			size += metaDataSize;
			size += VERSION_SIZE;
		}

		//
		for (Integer key : valueMap.keySet()) {
			LazySerializeType lazySerializeType = metaData.getColumnType(key);
			if (LazySerializeType.TYPE_UNKNOWN == lazySerializeType) {
				throw new RuntimeException("column index not fount:" + key);
			}

			Object value = valueMap.get(key);
			size += getColumnSize(lazySerializeType, key, value);
		}

		if (!isMetaDataFromOutside) {
			size += CodedOutputStream.computeInt32SizeNoTag(size);
		}

		return size;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		sb.append("{\n");

		Map<Integer, Object> map = new TreeMap<Integer, Object>(valueMap);
		for (Integer key : map.keySet()) {
			sb.append(key).append(":\t").append(map.get(key)).append('\n');
		}

		sb.append('}');

		return sb.toString();
	}

	@Override
	public DefaultLazyMessage mergeFrom(byte[] data) throws IOException {
		if (data.length == 0) {
			return this;
		}

		CodedInputStream input = CodedInputStream.newInstance(data);

		return mergeFrom(input);
	}

	@Override
	public DefaultLazyMessage mergeFrom(InputStream inputStream) throws IOException {
		CodedInputStream input = CodedInputStream.newInstance(inputStream);

		return mergeFrom(input);
	}

	@Override
	public DefaultLazyMessage mergeFrom(CodedInputStream input) throws IOException {
		//如果元数据不是外部传入的，则需要读取协议的长度、版本及元数据
		if (!isMetaDataFromOutside) {
			input.readInt32();	//total
			input.readInt32();	//version

			//读取元数据
			metaData = LazySerializeFactory.createLazyMetaData();
			input.readMessage(metaData);
		}

		while (true) {
			//读取元数据
			int tag = input.readTag();
			if (tag == 0) {
				return this;
			}

	        int columnIndex = WireFormat.getTagFieldNumber(tag);
	        LazySerializeType lazySerializeType = metaData.getColumnType(columnIndex);
	        switch (lazySerializeType) {
			case TYPE_INT:
				setInt(columnIndex, input.readInt32());
				break;
			case TYPE_LONG:
				setLong(columnIndex, input.readInt64());
				break;
			case TYPE_FLOAT:
				setFloat(columnIndex, input.readFloat());
				break;
			case TYPE_DOUBLE:
				setDouble(columnIndex, input.readDouble());
				break;
			case TYPE_BOOLEAN:
				setBoolean(columnIndex, input.readBool());
				break;
			case TYPE_STRING:
				setString(columnIndex, input.readString());
				break;
			case TYPE_BYTES:
				setBytes(columnIndex, input.readBytes());
				break;
			case TYPE_MESSAGE:
				ILazySerialize message = LazySerializeFactory.createLazyMessage();
				input.readMessage(message);
				setMessage(columnIndex, message);
				break;
			case TYPE_REPEAT:
				addAllToList(columnIndex, input.readRepeat());

				break;
			default:
				//忽略不识别的类型
				continue;
			}
		}
	}
}

