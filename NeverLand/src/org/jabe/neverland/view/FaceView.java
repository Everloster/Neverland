package org.jabe.neverland.view;

import java.util.ArrayList;
import java.util.List;

import org.jabe.neverland.R;

import android.content.Context;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.util.AttributeSet;
import android.view.Gravity;
import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemClickListener;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.LinearLayout;

public class FaceView extends LinearLayout {

    DragLayer mDrayLayer;
    PageDisplayView mProgressView;

    public ChooseFaceListener listener;

    private Context mContext;

    /** 总页数. */
    private int PageCount;
    /** 当前页码. */
    private int PageCurrent;
    /** 被选中的. */
    private int gridID = -1;
    /** 每页显示的数量，Adapter保持一致. */
    protected float mPageSize = 15.0f;
    /** GridView. */
    private GridView gridView;

    /** 数据集. */
    protected List<FaceInfo> mFaceInfos = new ArrayList<FaceInfo>();
    /** 页码条. */
    private ImageView imgCur;

    protected int mItemSize = 15;//每页显示表情个数

    protected int mNumColums = 5;

    protected int mHorizonSpace = 13;

    protected int mVerticalSpace = 13;

    private int mWidth;
    private int mHeight;

    public class FaceInfo {
        String name;
        int icon;

        public void setName(String name) {
            this.name = name;
        }

        public String getName() {
            return name;
        }

        public void setIcon(int icon) {
            this.icon = icon;
        }

        public int getIcon() {
            return icon;
        }
    }

    public interface ChooseFaceListener {
        void onChoosed(String text);
    }

    public void setFaceListener(ChooseFaceListener listener) {
        this.listener = listener;
    }

    public FaceView(Context context) {
        super(context);
    }

    public FaceView(Context context, AttributeSet attrs) {
        super(context, attrs);
        mContext = context;
    }

    @Override
    protected void onFinishInflate() {
        mDrayLayer = (DragLayer) findViewById(R.id.scroll_view);
        mProgressView = (PageDisplayView) findViewById(R.id.page_indicator);
        mDrayLayer.getLayoutParams().height = 200;
        setGrid();
        setCurPage(0);
        mDrayLayer.setPageListener(new DragLayer.PageListener() {

            @Override
            public void page(int page) {
                setCurPage(page);
            }
        });
    }

    /**
     * 设置当前页显示表情数量
     * @param size
     */
    protected void setPageSize(int size) {
        mPageSize = size;
    }

    protected void setNumClumn(int column) {
        mNumColums = column;
    }

    protected void setPaddingSpace(int padding) {
        mHorizonSpace = mVerticalSpace = padding;
    }

    protected void setItemSize(int itemSize) {
        mItemSize = itemSize;
    }

    protected void setLayoutParam(int width, int height) {
        mWidth = width;
        mHeight = height;
    }

    /**
     * 添加GridView
     */
    private void setGrid() {
        PageCount = (int) Math.ceil(mFaceInfos.size() / mPageSize);
        if (gridView != null) {
            mDrayLayer.removeAllViews();
        }
        for (int i = 0; i < PageCount; i++) {
            gridView = new GridView(mContext);
            gridView.setAdapter(new FaceViewAdapter(mContext, mFaceInfos, i, mItemSize).setLayoutParam(mWidth, mHeight));
            gridView.setNumColumns(mNumColums);
            gridView.setHorizontalSpacing(mHorizonSpace);
            gridView.setVerticalSpacing(mVerticalSpace);
            gridView.setGravity(Gravity.CENTER);
            //去掉点击时的黄色背景
            gridView.setSelector(new ColorDrawable(Color.TRANSPARENT));
            gridView.setOnItemClickListener(gridListener);
            mDrayLayer.addView(gridView);
        }
    }

    /**
     * GridView的监听事件
     */
    public OnItemClickListener gridListener = new OnItemClickListener() {

        public void onItemClick(AdapterView<?> arg0, View arg1, int arg2, long arg3) {
            PageCurrent = mDrayLayer.getCurScreen();
            gridID = arg2 + PageCurrent * mItemSize;
            if (null != listener) {
                listener.onChoosed(mFaceInfos.get(gridID).getName());
            }
            //            //            if (((GridView) arg0).getTag() != null) {
            //            //                ((View) ((GridView) arg0).getTag()).setBackgroundResource(R.drawable.bg_grid_item);
            //            //            }
            //            //            ((GridView) arg0).setTag(arg1);
            //            //            arg1.setBackgroundResource(R.drawable.bg_grid_item_false);

        }
    };

    /**
     * 更新当前页码
     */
    public void setCurPage(int page) {
        mProgressView.removeAllViews();
        for (int i = 0; i < PageCount; i++) {
            imgCur = new ImageView(mContext);
            imgCur.setBackgroundResource(R.drawable.progress_unchecked);
            imgCur.setId(i);
            // 判断当前页码来更新
            if (imgCur.getId() == page) {
                imgCur.setBackgroundResource(R.drawable.progress_checked);
            }
            mProgressView.addView(imgCur);
        }
    }

    public void setDefaultPage() {
        mDrayLayer.snapToScreen(0);
        setCurPage(0);
    }
}
