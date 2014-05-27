package org.jabe.neverland.graphics;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;

import javax.microedition.khronos.opengles.GL10;

public class Circle {

	//顶点坐标数据缓存
	private IntBuffer mVertexBuffer;
	//顶点法向量数据缓存
	private IntBuffer mNormalBuffer;
	//顶点构建索引数据缓存
//	private ByteBuffer mIndexBuffer;
	
	//顶点纹理数据缓冲
	private FloatBuffer mTextureBuffer;
	
	//沿x轴旋转角度
	public float mAngleX;
	//沿y轴旋转角度
	public float mAngleY;
	//沿z轴旋转角度
	public float mAngleZ;
	
	//顶点个数
	int vCount=0;
	//索引个数
//	int iCount=0;
	//纹理ID
	int textureId;
	
	/**
	 * 构造函数
	 * @param scale
	 */
	public Circle(int scale,int textureId){
		
		//设置纹理ID
		this.textureId = textureId;
		
		//单位值
		final int UNIT_SIZE = 10000;
		//将球进行单位切分的角度
		final int angleSpan = 18;
		//存放顶点坐标的集合
		ArrayList<Integer> alVertix = new ArrayList<Integer>();

		//垂直方向angleSpan度一份
		for(int vAngle = -90;vAngle <= 90;vAngle += angleSpan){
			//水平方向angleSpan度一份
			for(int hAngle=0;hAngle<360;hAngle+=angleSpan){
				//纵向横向各得到一个角度后计算对应的此点在球面上的坐标
				//计算椭球体上的长轴
				double xozLength=scale*UNIT_SIZE*Math.cos(Math.toRadians(vAngle));
				/**
				 * 计算球体的x坐标，球体的x坐标计算公式：x=a*cosn*cosm.
				 * n:维度
				 * m:经度
				 * a:长轴
				 */
        		int x=(int)(xozLength*Math.cos(Math.toRadians(hAngle)));
        		/**
        		 * 计算球体的y坐标：球体的y坐标计算公式：y=b*cosn*sinM.
        		 * n:维度
        		 * m:经度
        		 * b:短轴
        		 */
        		int z=(int)(xozLength*Math.sin(Math.toRadians(hAngle)));
        		/**
        		 * 计算球体的z坐标：球体的z坐标计算公式：z=c*sinn.
        		 * n:维度
        		 * c:纵轴
        		 */
        		int y=(int)(scale*UNIT_SIZE*Math.sin(Math.toRadians(vAngle)));
        		//将计算出来的XYZ坐标加入存放顶点坐标的ArrayList
        		alVertix.add(x);
        		alVertix.add(y);
        		alVertix.add(z);
			}
		}
		//顶点的数量为坐标值数量的1/3，因为一个顶点有3个坐标
		vCount = alVertix.size()/3;
		//将alVertix中的坐标值转存到一个int数组中
        int vertices[]=new int[vCount*3];
    	for(int i=0;i<alVertix.size();i++){
    		vertices[i]=alVertix.get(i);
    	}
    	
    	alVertix.clear();
    	//纹理
    	ArrayList<Float> alTexture = new ArrayList<Float>();
    	
    	int row=(180/angleSpan)+1;//球面切分的行数
        int col=360/angleSpan;//球面切分的列数
        for(int i=0;i<row;i++){//对每一行循环
        	if(i>0&&i<row-1){
        		//中间行
        		for(int j=-1;j<col;j++){
        			//中间行的两个相邻点与下一行的对应点构成三角形
					int k=i*col+j;
					//第一个三角形顶点
					alVertix.add(vertices[(k+col)*3]);
					alVertix.add(vertices[(k+col)*3+1]);
					alVertix.add(vertices[(k+col)*3+2]);					
					alTexture.add(0.0f);alTexture.add(0.0f);
					
					//第2个三角形顶点		
					alVertix.add(vertices[(k+1)*3]);
					alVertix.add(vertices[(k+1)*3+1]);
					alVertix.add(vertices[(k+1)*3+2]);					
					alTexture.add(1.0f);alTexture.add(1.0f);
					
					//第3个三角形顶点
					alVertix.add(vertices[k*3]);
					alVertix.add(vertices[k*3+1]);
					alVertix.add(vertices[k*3+2]);	
					alTexture.add(1.0f);alTexture.add(0.0f);
				}
        		for(int j=0;j<col+1;j++){
        			//中间行的两个相邻点与上一行的对应点构成三角形			
        			int k=i*col+j;
        			//第1个三角形顶点					
					alVertix.add(vertices[(k-col)*3]);
					alVertix.add(vertices[(k-col)*3+1]);
					alVertix.add(vertices[(k-col)*3+2]);					
					alTexture.add(1f);alTexture.add(1f);
					
					//第2个三角形顶点					
					alVertix.add(vertices[(k-1)*3]);
					alVertix.add(vertices[(k-1)*3+1]);
					alVertix.add(vertices[(k-1)*3+2]);					
					alTexture.add(0.0f);alTexture.add(0.0f);
					
					//第3个三角形顶点					
					alVertix.add(vertices[k*3]);
					alVertix.add(vertices[k*3+1]);
					alVertix.add(vertices[k*3+2]);					
					alTexture.add(0f);alTexture.add(1f);  
        		}
        	}
        }
        
        //顶点的数量为坐标值数量的1/3，因为一个顶点有3个坐标
        vCount=alVertix.size()/3;
        //将alVertix中的坐标值转存到一个int数组中
        vertices=new int[vCount*3];
    	for(int i=0;i<alVertix.size();i++)
    	{
    		vertices[i]=alVertix.get(i);
    	}
    	
    	//创建顶点坐标数据缓冲
        //vertices.length*4是因为一个整数四个字节
        ByteBuffer vbb = ByteBuffer.allocateDirect(vertices.length*4);
        vbb.order(ByteOrder.nativeOrder());//设置字节顺序
        mVertexBuffer = vbb.asIntBuffer();//转换为int型缓冲
        mVertexBuffer.put(vertices);//向缓冲区中放入顶点坐标数据
        mVertexBuffer.position(0);//设置缓冲区起始位置     
        //创建顶点法向量数据缓冲
        //vertices.length*4是因为一个float四个字节
        ByteBuffer nbb = ByteBuffer.allocateDirect(vertices.length*4);
        nbb.order(ByteOrder.nativeOrder());//设置字节顺序
        mNormalBuffer = vbb.asIntBuffer();//转换为int型缓冲
        mNormalBuffer.put(vertices);//向缓冲区中放入顶点坐标数据
        mNormalBuffer.position(0);//设置缓冲区起始位置
        
        //创建纹理坐标缓冲
        float textureCoors[]=new float[alTexture.size()];//顶点纹理值数组
        for(int i=0;i<alTexture.size();i++) 
        {
        	textureCoors[i]=alTexture.get(i);
        }
        
        ByteBuffer cbb = ByteBuffer.allocateDirect(textureCoors.length*4);
        cbb.order(ByteOrder.nativeOrder());//设置字节顺序
        mTextureBuffer = cbb.asFloatBuffer();//转换为int型缓冲
        mTextureBuffer.put(textureCoors);//向缓冲区中放入顶点着色数据
        mTextureBuffer.position(0);//设置缓冲区起始位置
	}
	
	public void drawSelf(GL10 gl){
    	gl.glRotatef(mAngleZ, 0, 0, 1);//沿Z轴旋转
    	gl.glRotatef(mAngleX, 1, 0, 0);//沿X轴旋转
        gl.glRotatef(mAngleY, 0, 1, 0);//沿Y轴旋转
        
        //移动转换
        gl.glTranslatef(DataConfig.moveX, DataConfig.moveY, 0); 
        
        //缩放变换
        gl.glScalef(DataConfig.scaleX, DataConfig.scaleY, DataConfig.scaleZ);
        
        //允许使用顶点数组
        gl.glEnableClientState(GL10.GL_VERTEX_ARRAY);
        
		//为画笔指定顶点坐标数据
        gl.glVertexPointer(
        		3,				//每个顶点的坐标数量为3  xyz 
        		GL10.GL_FIXED,	//顶点坐标值的类型为 GL_FIXED
        		0, 				//连续顶点坐标数据之间的间隔
        		mVertexBuffer	//顶点坐标数据
        );
       
        //允许使用法向量数组
        gl.glEnableClientState(GL10.GL_NORMAL_ARRAY);
        //为画笔指定顶点法向量数据
        gl.glNormalPointer(GL10.GL_FIXED, 0, mNormalBuffer);
		
        //开启纹理
        gl.glEnable(GL10.GL_TEXTURE_2D);   
        //允许使用纹理ST坐标缓冲
        gl.glEnableClientState(GL10.GL_TEXTURE_COORD_ARRAY);
        //为画笔指定纹理ST坐标缓冲
        gl.glTexCoordPointer(2, GL10.GL_FLOAT, 0, mTextureBuffer);
        //绑定当前纹理
        gl.glBindTexture(GL10.GL_TEXTURE_2D, textureId);
        
        //绘制图形
        gl.glDrawArrays
        (
        		GL10.GL_TRIANGLES, 		//以三角形方式填充
        		0, 			 			//开始点编号
        		vCount					//顶点数量
        ); 
	}
}
