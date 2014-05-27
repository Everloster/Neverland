package org.jabe.neverland.view;

import java.util.ArrayList;
import java.util.List;

import org.jabe.neverland.view.FaceView.FaceInfo;

import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.GridView;
import android.widget.ImageView;

public class FaceViewAdapter extends BaseAdapter {

    private Context mContext;
    /** 列表. */
    private List<FaceInfo> mFaceInfos;

    private int mSize;

    private int mWidth;
    private int mHeight;

    public FaceViewAdapter(Context context, List<FaceInfo> faceInfos, int page, int size) {
        mContext = context;
        mFaceInfos = new ArrayList<FaceInfo>();
        mSize = size;
        int i = page * size;
        int iEnd = i + size;
        while ((i < faceInfos.size()) && (i < iEnd)) {
            mFaceInfos.add(faceInfos.get(i));
            i++;
        }
    }

    public FaceViewAdapter setLayoutParam(int width, int height) {
        mWidth = width;
        mHeight = height;
        return this;
    }

    public int getSize(int size) {// 每页显示的Item个数
        return mSize;
    }

    @Override
    public int getCount() {
        return mFaceInfos.size();
    }

    @Override
    public Object getItem(int position) {
        return mFaceInfos.get(position);
    }

    @Override
    public long getItemId(int position) {
        return position;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        ImageView face = new ImageView(mContext);
        face.setLayoutParams(new GridView.LayoutParams(mWidth, mHeight));
        face.setBackgroundResource(mFaceInfos.get(position).getIcon());
        return face;
    }

}
