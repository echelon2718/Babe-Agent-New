from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    Float,
    DECIMAL,
    Boolean,
    JSON,
    DateTime,
    ForeignKey,
    Table,
    Time,
    Enum,
    TIMESTAMP,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from .sqlalchemy_setup import Base
from sqlalchemy.sql import func

combo_product = Table(
    "combo_product",
    Base.metadata,
    Column("combo_id", ForeignKey("combos.id"), primary_key=True),
    Column("product_id", ForeignKey("products.id"), primary_key=True),
    Column("item_id",Integer,nullable=True),
    Column("olsera_prod_id",BigInteger,nullable=True),
    Column("olsera_combo_id",BigInteger,nullable=True),
    Column("qty", Integer, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)





outlet_condition = Table(
    "outlet_condition",
    Base.metadata,
    Column("outlet_id", BigInteger, ForeignKey("outlets.id"), primary_key=True),
    Column("condition_id", BigInteger, ForeignKey("conditions.id"), primary_key=True),
)


class Combo(Base):
    __tablename__ = "combos"

    id = Column(Integer, primary_key=True)
    olsera_id = Column(Integer)
    outlet_id = Column(Integer)
    name = Column(String(255))
    description = Column(String(255))
    image = Column(String(255))
    price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    products = relationship("Product", secondary=combo_product, back_populates="combos")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    olsera_id = Column(Integer)
    outlet_id = Column(Integer)
    name = Column(String(255))
    klasifikasi_id = Column(BigInteger)
    klasifikasi = Column(String(255))
    image = Column(String(255))
    description = Column(Text)
    tag = Column(String(255))
    alias = Column(String(255))
    percent_alkohol = Column(DECIMAL(5, 2))
    keywords = Column(String(255))
    price = Column(DECIMAL(10, 2))
    koin = Column(BigInteger)
    has_variant = Column(Boolean)
    variants = Column(JSON)
    total_dibeli = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # relasionships
    stock = relationship("ProductStock", uselist=False, back_populates="product")
    combos = relationship("Combo", secondary=combo_product, back_populates="products")


class ProductStock(Base):
    __tablename__ = "product_stocks"

    product_id = Column(Integer, ForeignKey("products.id"), primary_key=True)
    stock_qty = Column(Integer)
    product = relationship("Product", back_populates="stock")


class Cart(Base):
    __tablename__ = "carts"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, nullable=False)
    outlet_id = Column(BigInteger, ForeignKey("outlets.id"), nullable=False)

    product_id = Column(String(255))
    prodvar_id = Column(String(255))
    product_type_id = Column(BigInteger, nullable=False)

    name = Column(String(255))
    combo_id = Column(String(255))
    bundle_id = Column(BigInteger)

    klasifikasi_id = Column(String(255))
    klasifikasi = Column(String(255))
    variant_id = Column(String(255))
    variant = Column(String(255))

    harga_satuan = Column(BigInteger)
    discount = Column(DECIMAL(10, 3), default=0.000)

    quantity = Column(Integer)
    harga_total = Column(BigInteger)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # relationships
    outlet = relationship("Outlet", back_populates="carts")


class Outlet(Base):
    __tablename__ = "outlets"

    id = Column(BigInteger, primary_key=True)
    name = Column(String(255), nullable=False)

    jam_buka = Column(Time)  # tipe time
    jam_tutup = Column(Time)  # tipe time

    latitude = Column(DECIMAL(10, 8))  # akurasi tinggi untuk koordinat
    longitude = Column(DECIMAL(11, 8))  # sesuai dengan struktur tabel

    address = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relasi ke cart
    carts = relationship("Cart", back_populates="outlet")
    conditions = relationship(
        "Condition", secondary=outlet_condition, back_populates="outlets"
    )
    phone = Column(String(50), nullable=True, unique=True)
    order_histories = relationship("Order", back_populates="outlet")


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True, unique=True)
    jumlah_koin = Column(Integer, nullable=False, default=0)
    free_instant_delivery = Column(Integer, nullable=False, default=0)
    phone = Column(String(255), nullable=True, unique=True)
    address = Column(String(255), nullable=True)
    latitude = Column(DECIMAL(10, 8), nullable=True)
    longitude = Column(DECIMAL(11, 8), nullable=True)
    password = Column(String(255), nullable=False)
    remember_token = Column(String(100), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class Condition(Base):
    __tablename__ = "conditions"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255))
    nilai = Column(Integer, nullable=False)
    outlets = relationship(
        "Outlet", secondary=outlet_condition, back_populates="conditions"
    )


class Order(Base):
    __tablename__ = "order_histories"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    outlet_id = Column(BigInteger, ForeignKey("outlets.id"), nullable=False)
    delivery_id = Column(BigInteger, ForeignKey("deliveries.id"), nullable=False)
    order_id = Column(BigInteger, nullable=False)
    order_no = Column(String(255), nullable=False)
    items = Column(JSON, nullable=False)
    subtotal = Column(Integer, nullable=False)
    shipping_fee = Column(Integer, nullable=False)
    tax = Column(Integer, nullable=False)
    discount_amount = Column(Integer, nullable=False, default=0)
    voucher_code = Column(String(255), nullable=True)
    total = Column(Integer, nullable=False)
    koin = Column(Integer, nullable=False, default=0)
    distance_km = Column(Float, nullable=False)
    delivery_address = Column(String(255), nullable=False)
    delivery_latitude = Column(DECIMAL(10, 8), nullable=True)
    delivery_longitude = Column(DECIMAL(11, 8), nullable=True)
    payment_type = Column(String(255), nullable=False, default="QRIS")
    payment_status = Column(
        Enum(
            "pending",
            "success",
            "failed",
            "expired",
            "canceled",
            name="payment_status_enum",
        ),
        nullable=False,
        default="pending",
    )
    notes = Column(Text, nullable=True)
    estimasi_tiba = Column(Time, nullable=True)
    created_at = Column(TIMESTAMP, nullable=True)
    updated_at = Column(TIMESTAMP, nullable=True)

    # Relasi
    outlet = relationship("Outlet", back_populates="order_histories")


class StrukLog(Base):
    __tablename__ = "struk_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    order_id = Column(BigInteger, nullable=False)
    order_no = Column(String(255), nullable=False)
    is_forward = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())