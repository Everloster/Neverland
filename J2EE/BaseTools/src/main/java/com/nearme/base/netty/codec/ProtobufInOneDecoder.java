/**
 * ProtobufInOneDecoder.java
 * com.nearme.base.netty.codec
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-12-24 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.codec;

import java.net.SocketAddress;
import java.util.ArrayList;
import java.util.List;

import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.buffer.ChannelBufferInputStream;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.channel.ChannelStateEvent;
import org.jboss.netty.channel.MessageEvent;
import org.jboss.netty.handler.codec.frame.CorruptedFrameException;
import org.jboss.netty.handler.codec.frame.FrameDecoder;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.MessageLite;

/**
 * ClassName:ProtobufInOneDecoder <br>
 * Function: pb类型数据解码，整合ProtobufVarint32FrameDecoder与ProtobufDecoder。 <br>
 * mergeRequest设置为true，可将多个请求的pb数据合并到一个List中统一处理，减少notify()次数
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-12-24  上午10:29:18
 */
public class ProtobufInOneDecoder extends FrameDecoder {
	private final MessageLite prototype;
    private final ExtensionRegistry extensionRegistry;
    private final boolean mergeRequest;
	
    public ProtobufInOneDecoder(MessageLite prototype) {
        this(prototype, null);
    }
    
    public ProtobufInOneDecoder(MessageLite prototype, boolean mergeRequest) {
        this(prototype, null, mergeRequest);
    }
    
    public ProtobufInOneDecoder(MessageLite prototype, ExtensionRegistry extensionRegistry) {
    	this(prototype, null, true);
    }

    public ProtobufInOneDecoder(MessageLite prototype, ExtensionRegistry extensionRegistry, boolean mergeRequest) {
    	super();
    	
        if (prototype == null) {
            throw new NullPointerException("prototype");
        }
        
        this.prototype = prototype.getDefaultInstanceForType();
        this.extensionRegistry = extensionRegistry;
        this.mergeRequest = mergeRequest;
    }
	
	@Override
    public void messageReceived(ChannelHandlerContext ctx, MessageEvent e) throws Exception {
        Object m = e.getMessage();
        if (!(m instanceof ChannelBuffer)) {
            ctx.sendUpstream(e);
            return;
        }

        ChannelBuffer input = (ChannelBuffer) m;
        if (!input.readable()) {
            return;
        }

        if (cumulation == null) {
            try {
                // the cumulation buffer is not created yet so just pass the input to callDecode(...) method
                callDecode0(ctx, e.getChannel(), input, e.getRemoteAddress());
            } finally {
                updateCumulation(ctx, input);
            }
        } else {
            input = appendToCumulation(input);
            try {
                callDecode0(ctx, e.getChannel(), input, e.getRemoteAddress());
            } finally {
                updateCumulation(ctx, input);
            }
        }
    }
	
	private void callDecode0(
            ChannelHandlerContext context, Channel channel,
            ChannelBuffer cumulation, SocketAddress remoteAddress) throws Exception {
		//将多个请求的数据合并到一个中，然后统一处理，减少notify()次数
		List<Object> results = null;
        while (cumulation.readable()) {
            int oldReaderIndex = cumulation.readerIndex();
            Object frame = decode(context, channel, cumulation);
            if (frame == null) {
                if (oldReaderIndex == cumulation.readerIndex()) {
                    // Seems like more data is required.
                    // Let us wait for the next notification.
                    break;
                } else {
                    // Previous data has been discarded.
                    // Probably it is reading on.
                    continue;
                }
            } else if (oldReaderIndex == cumulation.readerIndex()) {
                throw new IllegalStateException(
                        "decode() method must read at least one byte " +
                        "if it returned a frame (caused by: " + getClass() + ")");
            }

            if(mergeRequest) {
	            if(null == results) {
	            	results = new ArrayList<Object>();
	            }
	            results.add(frame);
            } else {
            	unfoldAndFireMessageReceived(context, remoteAddress, frame);
            }
        }
        
        if(mergeRequest && null != results) {
        	//System.out.println("ProtobufInOneDecoder size:" + results.size());
        	unfoldAndFireMessageReceived(context, remoteAddress, results);
        }
    }
	
	@Override
    protected Object decode(ChannelHandlerContext ctx, Channel channel, ChannelBuffer buffer) throws Exception {
		ChannelBuffer buf = decodePbOriginalBuffer(ctx, channel, buffer);
		if(null == buf) {
			return null;
		}
		
		if (buf.hasArray()) {
            final int offset = buf.readerIndex();
            if (extensionRegistry == null) {
                return prototype.newBuilderForType().mergeFrom(
                        buf.array(), buf.arrayOffset() + offset, buf.readableBytes()).build();
            } else {
                return prototype.newBuilderForType().mergeFrom(
                        buf.array(), buf.arrayOffset() + offset, buf.readableBytes(), extensionRegistry).build();
            }
        } else {
            if (extensionRegistry == null) {
                return prototype.newBuilderForType().mergeFrom(
                        new ChannelBufferInputStream(buf)).build();
            } else {
                return prototype.newBuilderForType().mergeFrom(
                        new ChannelBufferInputStream(buf), extensionRegistry).build();
            }
        }
    }
	
	//解析得到原始的pb数据（二进制）
	private ChannelBuffer decodePbOriginalBuffer(ChannelHandlerContext ctx, Channel channel, ChannelBuffer buffer) throws Exception {
		buffer.markReaderIndex();
        final byte[] buf = new byte[5];
        for (int i = 0; i < buf.length; i ++) {
            if (!buffer.readable()) {
                buffer.resetReaderIndex();
                return null;
            }

            buf[i] = buffer.readByte();
            if (buf[i] >= 0) {
                int length = CodedInputStream.newInstance(buf, 0, i + 1).readRawVarint32();
                if (length < 0) {
                    throw new CorruptedFrameException("negative length: " + length);
                }

                if (buffer.readableBytes() < length) {
                    buffer.resetReaderIndex();
                    return null;
                } else {
                    return buffer.readBytes(length);
                }
            }
        }

        // Couldn't find the byte whose MSB is off.
        throw new CorruptedFrameException("length wider than 32-bit");
	}
	
	/**
     * Gets called on {@link #channelDisconnected(ChannelHandlerContext, ChannelStateEvent)} and
     * {@link #channelClosed(ChannelHandlerContext, ChannelStateEvent)}
     */
    protected void cleanup(ChannelHandlerContext ctx, ChannelStateEvent e)
            throws Exception {
        try {
            ChannelBuffer cumulation = this.cumulation;
            if (cumulation == null) {
                return;
            }

            this.cumulation = null;

            if (cumulation.readable()) {
                // Make sure all frames are read before notifying a closed channel.
                callDecode0(ctx, ctx.getChannel(), cumulation, null);
            }

            // Call decodeLast() finally.  Please note that decodeLast() is
            // called even if there's nothing more to read from the buffer to
            // notify a user that the connection was closed explicitly.
            Object partialFrame = decodeLast(ctx, ctx.getChannel(), cumulation);
            if (partialFrame != null) {
                unfoldAndFireMessageReceived(ctx, null, partialFrame);
            }
        } finally {
            ctx.sendUpstream(e);
        }
    }
}

